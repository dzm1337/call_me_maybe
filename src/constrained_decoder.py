from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model  # type: ignore
from src.trie import Trie
from src.models import FunctionCallResult, FunctionDef, ParamType
from src.vocabulary import load_vocabulary
from src.token_decoder import TokenDecoder

NEG_INF: float = float("-inf")
FloatArray: TypeAlias = npt.NDArray[np.float64]


class ConstrainedDecoder:
    """
    Orchestrates structured constrained decoding.

    Responsible for selecting a valid function name and generating
    schema-compliant parameters while preserving JSON structure.
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDef],
    ) -> None:
        """
        Initialize decoding components and structural helpers.
        """

        # Store model reference and function schema.
        self.model = model
        self.functions = functions
        self.fn_names = [f.name for f in functions]
        self.fn_map = {f.name: f for f in functions}

        # Load tokenizer vocabulary for token-to-text mapping.
        self.id_to_token, self.token_to_id = load_vocabulary(model)

        # Precompute structural token categories.
        self.stop_ids = self._load_stop_ids()
        self.quote_ids = self._load_quote_ids()
        self.newline_ids = self._load_newline_ids()

        # Trie enforces prefix-constrained function name decoding.
        self.fn_trie = Trie()
        for name in self.fn_names:
            self.fn_trie.insert(name)

        # TokenDecoder handles primitive value generation.
        self.token_decoder = TokenDecoder(
            model=self.model,
            id_to_token=self.id_to_token,
            stop_ids=self.stop_ids,
            newline_ids=self.newline_ids,
        )

    def generate(self, user_prompt: str) -> FunctionCallResult:
        """
        Execute full constrained decoding pipeline.

        Produces a valid function call strictly aligned
        with the declared function schema.
        """
        prompt = self._build_prompt(user_prompt)
        context = self._encode(prompt)

        fn_name, context = self._generate_fn_name(context)
        fn_def = self.fn_map[fn_name]

        parameters, context = self._generate_parameters(
            context,
            fn_def,
            user_prompt,
        )

        return FunctionCallResult(
            prompt=user_prompt,
            name=fn_name,
            parameters=parameters,
        )

    def _generate_fn_name(
        self,
        context: list[int],
    ) -> tuple[str, list[int]]:
        """
        Decode a function name under trie-based masking.

        The model is only allowed to extend valid prefixes,
        ensuring the final output matches a declared function.
        """
        generated = ""
        current_context = list(context)

        for _ in range(15):
            logits = self._get_logits(current_context)
            masked = self._mask_for_fn_name(logits, generated)

            next_id = int(np.argmax(masked))
            token_str = self.id_to_token.get(next_id, "")
            clean_token = self._clean(token_str)

            current_context.append(next_id)
            generated += clean_token

            if self.fn_trie.is_complete_word(generated):
                return generated, current_context

            if not self.fn_trie.is_prefix(generated):
                break

        return self._best_fn_match(generated), current_context

    def _mask_for_fn_name(
        self,
        logits: FloatArray,
        current: str,
    ) -> FloatArray:
        """
        Restrict tokens that would invalidate the current trie prefix.

        Only continuations that maintain a valid function name
        remain selectable.
        """
        masked = np.full_like(logits, NEG_INF)

        for tid, token_str in self.id_to_token.items():
            clean = self._clean(token_str)
            if not clean:
                continue

            candidate = current + clean

            if self.fn_trie.is_prefix(candidate) or self.fn_trie.is_complete_word(
                candidate
            ):
                masked[tid] = logits[tid]

        return masked if np.any(masked != NEG_INF) else logits.copy()

    def _best_fn_match(self, text: str) -> str:
        """Fallback to closest matching function name."""
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
        user_prompt: str,
    ) -> tuple[dict[str, Any], list[int]]:
        """
        Decode parameters according to declared types.

        Primitive values are delegated to TokenDecoder.
        """
        context = context + self._encode('", "parameters": {')
        params: dict[str, Any] = {}
        value: Any

        for i, (param_name, param_typedef) in enumerate(fn_def.parameters.items()):
            separator = "" if i == 0 else ", "
            context += self._encode(f'{separator}"{param_name}":')

            param_type = param_typedef.type

            if param_type in {ParamType.NUMBER, ParamType.INTEGER}:
                value, context = self.token_decoder.gen_number(context, param_type)

            elif param_type == ParamType.BOOLEAN:
                value, context = self.token_decoder.gen_boolean(context)

            else:
                context += self._encode('"')
                value, context = self.token_decoder.gen_string(context)
                context += self._encode('"')

            params[param_name] = value

        context += self._encode("}}")
        params = self._normalize_parameters(params, fn_def)

        return params, context

    def _normalize_parameters(
        self,
        params: dict[str, Any],
        fn_def: FunctionDef,
    ) -> dict[str, Any]:
        """Ensure generated parameters match declared types."""
        params = params.copy()

        for name, typedef in fn_def.parameters.items():
            if name not in params:
                continue

            value = params[name]
            param_type = typedef.type

            if param_type == ParamType.INTEGER and not isinstance(value, int):
                params[name] = int(value)

            if param_type == ParamType.NUMBER and not isinstance(value, float):
                params[name] = float(value)

        return params

    def _build_prompt(self, user_prompt: str) -> str:
        """Construct the original system prompt."""
        fn_lines = []
        for f in self.functions:
            params = ", ".join(
                f"{pname}: {pdef.type.value}" for pname, pdef in f.parameters.items()
            )
            fn_lines.append(f"- {f.name}({params}): {f.description}")
        fn_list = "\n".join(fn_lines)
        return (
            "You are a function calling assistant.\n"
            "Read the user request carefully and select the single most "
            "appropriate function.\n"
            "The function name must exactly match one of the available "
            "functions.\n"
            "Extract argument values exactly as they appear in the user "
            "request, preserving the original spelling and casing.\n\n"
            f"Available functions:\n{fn_list}\n\n"
            f"User request: {user_prompt}\n"
            'Selected function name: "'
        )

    def _clean(self, token_str: str) -> str:
        """Remove tokenizer spacing markers."""
        return token_str.replace("Ġ", " ").replace("Ċ", "\n")

    def _encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        result = self.model.encode(text)

        if hasattr(result, "tolist"):
            result = result.tolist()

        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]

        return [int(x) for x in result]

    def _get_logits(self, token_ids: list[int]) -> FloatArray:
        """Return logits for next-token prediction."""
        raw = self.model.get_logits_from_input_ids(token_ids)

        while isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]

        return np.array(raw, dtype=np.float64).flatten()

    def _load_stop_ids(self) -> set[int]:
        """Identify model termination tokens."""
        stop_strings = {
            "<|endoftext|>",
            "</s>",
            "<|im_end|>",
            "<eos>",
            "<|eot_id|>",
        }
        return {tid for tid, tok in self.id_to_token.items() if tok in stop_strings}

    def _load_quote_ids(self) -> set[int]:
        """Identify tokens representing double quotes."""
        return {tid for tid, tok in self.id_to_token.items() if self._clean(tok) == '"'}

    def _load_newline_ids(self) -> set[int]:
        """Identify tokens representing newline characters."""
        return {
            tid for tid, tok in self.id_to_token.items() if "Ċ" in tok or "\n" in tok
        }
