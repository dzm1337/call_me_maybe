import re
from typing import Any

import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model

from src.models import FunctionCallResult, FunctionDef, ParamType
from src.vocabulary import load_vocabulary

NEG_INF: float = float("-inf")


class ConstrainedDecoder:
    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDef],
    ) -> None:
        """Initialize decoder, loading vocabulary and lookup tables."""
        self.model = model
        self.functions = functions
        self.fn_names: list[str] = [f.name for f in functions]
        self.fn_map: dict[str, FunctionDef] = {f.name: f for f in functions}
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self.id_to_token, self.token_to_id = load_vocabulary(model)

        self.stop_ids: set[int] = self._load_stop_ids()
        self.quote_ids: set[int] = self._load_quote_ids()
        self.newline_ids: set[int] = self._load_newline_ids()

    def _load_stop_ids(self) -> set[int]:
        stop_strings = {
            "<|endoftext|>",
            "</s>",
            "<|im_end|>",
            "<eos>",
            "<|eot_id|>",
        }
        return {
            tid for tid, tok in self.id_to_token.items() if tok in stop_strings
        }

    def _load_quote_ids(self) -> set[int]:
        return {
            tid
            for tid, tok in self.id_to_token.items()
            if self._clean(tok) == '"'
        }

    def _load_newline_ids(self) -> set[int]:
        return {
            tid
            for tid, tok in self.id_to_token.items()
            if "Ċ" in tok or "\n" in tok
        }

    def generate(self, user_prompt: str) -> FunctionCallResult:
        fn_name = self._select_function(user_prompt)
        fn_def = self.fn_map[fn_name]
        parameters = self._extract_parameters(user_prompt, fn_def)
        return FunctionCallResult(
            prompt=user_prompt,
            name=fn_name,
            parameters=parameters,
        )

    def _build_prompt(self, user_prompt: str) -> str:
        fn_lines = []
        for f in self.functions:
            params = ", ".join(
                f"{pname}: {pdef.type.value}"
                for pname, pdef in f.parameters.items()
            )
            fn_lines.append(f"- {f.name}({params}): {f.description}")
        fn_list = "\n".join(fn_lines)

        return (
            "You are a function calling assistant.\n"
            "Read the user request carefully and select the single most appropriate function.\n"
            "The function name must exactly match one of the available functions.\n\n"
            f"Available functions:\n{fn_list}\n\n"
            f"User request: {user_prompt}\n"
            'Selected function name: "'
        )

    def _generate_fn_name(self, context: list[int]) -> tuple[str, list[int]]:
        generated = ""

        for _ in range(80):
            logits = self._get_logits(context)
            masked = self._mask_for_fn_name(logits, generated)
            next_id = int(np.argmax(masked))
            token_str = self.id_to_token.get(next_id, "")

            context = context + [next_id]
            generated += token_str

            clean = self._clean(generated)
            if clean in self.fn_names:
                return clean, context

            if not self._is_fn_prefix(clean):
                break

        best = self._best_fn_match(self._clean(generated))
        return best, context

    def _mask_for_fn_name(
        self,
        logits: np.ndarray,
        current: str,
    ) -> np.ndarray:
        masked = np.full_like(logits, NEG_INF)
        current_clean = self._clean(current)

        for token_id, token_str in self.id_to_token.items():
            candidate = current_clean + self._clean(token_str)
            if self._is_fn_prefix(candidate) or candidate in self.fn_names:
                masked[token_id] = logits[token_id]

        if np.all(masked == NEG_INF):
            return logits.copy()

        return masked

    def _is_fn_prefix(self, text: str) -> bool:
        return any(name.startswith(text) for name in self.fn_names)

    def _best_fn_match(self, text: str) -> str:
        best_name = self.fn_names[0]
        best_len = -1
        for name in self.fn_names:
            common = sum(1 for a, b in zip(text, name) if a == b)
            if common > best_len:
                best_len = common
                best_name = name
        return best_name

    def _generate_parameters(
        self,
        context: list[int],
        fn_def: FunctionDef,
    ) -> tuple[dict[str, Any], list[int]]:
        context = context + self._encode('", "parameters": {')

        params: dict[str, Any] = {}
        param_items = list(fn_def.parameters.items())

        for i, (param_name, param_typedef) in enumerate(param_items):
            separator = "" if i == 0 else ", "
            context = context + self._encode(f'{separator}"{param_name}": ')

            param_type = param_typedef.type

            if param_type == ParamType.NUMBER:
                value, context = self._gen_number(context)
            elif param_type == ParamType.BOOLEAN:
                value, context = self._gen_boolean(context)
            else:
                context = context + self._encode('"')
                value, context = self._gen_string(context)
                context = context + self._encode('"')

            params[param_name] = value

        context = context + self._encode("}}")
        return params, context

    def _gen_number(self, context: list[int]) -> tuple[float, list[int]]:
        generated = ""
        current_context = list(context)

        for _ in range(15):
            logits = self._get_logits(current_context)
            masked = self._mask_for_number(logits, generated)
            next_id = int(np.argmax(masked))
            token_str = self._clean(self.id_to_token.get(next_id, ""))

            if not token_str:
                break

            candidate = generated + token_str
            if not self._is_number_prefix(candidate):
                break

            current_context = current_context + [next_id]
            generated = candidate

            if self._is_complete_number(generated):
                next_logits = self._get_logits(current_context)
                next_id2 = int(np.argmax(next_logits))
                next_str = self._clean(self.id_to_token.get(next_id2, ""))
                if not self._would_extend_number(next_str, generated):
                    break

        return self._parse_number(generated), current_context

    def _mask_for_number(
        self,
        logits: np.ndarray,
        current: str,
    ) -> np.ndarray:
        masked = np.full_like(logits, NEG_INF)

        for token_id, token_str in self.id_to_token.items():
            text = self._clean(token_str)
            if not text:
                continue
            candidate = current + text
            if self._is_number_prefix(candidate) and len(candidate) <= 10:
                masked[token_id] = logits[token_id]

        if np.all(masked == NEG_INF):
            return logits.copy()

        return masked

    def _is_number_prefix(self, text: str) -> bool:
        if not text:
            return True
        return bool(re.match(r"^-?\d*\.?\d*$", text))

    def _is_complete_number(self, text: str) -> bool:
        if not text:
            return False
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _would_extend_number(self, next_str: str, current: str) -> bool:
        if not next_str:
            return False
        candidate = current + next_str
        if not self._is_number_prefix(candidate):
            return False
        if len(candidate) > 10:
            return False
        return True

    def _parse_number(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            match = re.search(r"-?\d+\.?\d*", text)
            if match:
                return float(match.group())
            return 0.0

    def _gen_boolean(self, context: list[int]) -> tuple[bool, list[int]]:
        logits = self._get_logits(context)
        true_score: float = NEG_INF
        false_score: float = NEG_INF
        true_id: int | None = None
        false_id: int | None = None

        for token_id, token_str in self.id_to_token.items():
            text = self._clean(token_str).lower()
            score = float(logits[token_id])
            if text == "true" and score > true_score:
                true_score = score
                true_id = token_id
            elif text == "false" and score > false_score:
                false_score = score
                false_id = token_id

        if true_score >= false_score and true_id is not None:
            return True, context + [true_id]
        if false_id is not None:
            return False, context + [false_id]
        return False, context

    def _gen_string(
        self,
        context: list[int],
    ) -> tuple[str, list[int]]:
        parts: list[str] = []
        current_context = list(context)

        for _ in range(80):
            logits = self._get_logits(current_context)
            next_id = int(np.argmax(logits))
            token_raw = self.id_to_token.get(next_id, "")

            if next_id in self.stop_ids:
                break

            if next_id in self.newline_ids:
                break

            decoded = token_raw.replace("Ġ", " ").replace("Ċ", "\n")

            if '"' in decoded:
                before_quote = decoded.split('"', 1)[0]
                if before_quote:
                    parts.append(before_quote)
                break

            parts.append(decoded)
            current_context = current_context + [next_id]

        result = "".join(parts).strip()
        return result, current_context

    def _mask_for_string(self, logits: np.ndarray) -> np.ndarray:
        masked = logits.copy()

        for token_id, token_str in self.id_to_token.items():
            decoded = token_str.replace("Ġ", " ").replace("Ċ", "\n")

            if decoded.startswith('"'):
                masked[token_id] = NEG_INF
            elif "\\" in decoded:
                masked[token_id] = NEG_INF
            elif decoded.strip() and all(
                c in "{}[]," for c in decoded.strip()
            ):
                masked[token_id] = NEG_INF

        return masked

    def _clean(self, token_str: str) -> str:
        return token_str.replace("Ġ", " ").replace("Ċ", "\n").strip()

    def _encode(self, text: str) -> list[int]:
        result = self.model.encode(text)
        if hasattr(result, "tolist"):
            result = result.tolist()

        if (
            isinstance(result, list)
            and len(result) > 0
            and isinstance(result[0], list)
        ):
            result = result[0]

        return [int(x) for x in result]

    def _get_logits(self, token_ids: list[int]) -> np.ndarray:
        raw = self.model.get_logits_from_input_ids(token_ids)

        if hasattr(raw, "tolist"):
            raw = raw.tolist()

        while (
            isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list)
        ):
            raw = raw[0]

        return np.array(raw, dtype=np.float64).flatten()
