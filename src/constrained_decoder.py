import re
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model  # type: ignore
from src.models import FunctionCallResult, FunctionDef, ParamType
from src.vocabulary import load_vocabulary

NEG_INF: float = float("-inf")
# Convenience alias for a 1D float64 numpy array used as logits
FloatArray: TypeAlias = npt.NDArray[np.float64]


class TrieNode:
    """A single node in the Trie,
    holding its children and end-of-word flag."""

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False


class Trie:
    """Prefix tree used to efficiently constrain
    decoding to valid function names."""

    def __init__(self) -> None:
        self.root: TrieNode = TrieNode()

    def insert(self, word: str) -> None:
        """Insert a word character by character into the trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def get_allowed_next_chars(self, prefix: str) -> set[str]:
        """Return the set of valid next
        characters that extend the given prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return set()
            node = node.children[char]
        return set(node.children.keys())

    def is_prefix(self, text: str) -> bool:
        """Return True if text is a valid prefix of any word in the trie."""
        node = self.root
        for char in text:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def is_complete_word(self, text: str) -> bool:
        """Return True if text exactly matches a fully inserted word."""
        node = self.root
        for char in text:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word


class ConstrainedDecoder:
    """
    Decodes structured function calls from an LLM using constrained token masking.
    Ensures the model only produces valid function names and typed parameters.
    """

    def __init__(
        self, model: Small_LLM_Model, functions: list[FunctionDef]
    ) -> None:
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

        # Build the trie from all registered function names for prefix-based masking
        self.fn_trie = Trie()
        for name in self.fn_names:
            self.fn_trie.insert(name)

    def _load_stop_ids(self) -> set[int]:
        """Collect token IDs that signal the end of a sequence."""
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
        """Collect token IDs whose cleaned
        representation is a double-quote character."""
        return {
            tid
            for tid, tok in self.id_to_token.items()
            if self._clean(tok) == '"'
        }

    def _load_newline_ids(self) -> set[int]:
        """Collect token IDs that represent newline or line-break characters."""
        return {
            tid
            for tid, tok in self.id_to_token.items()
            if "Ċ" in tok or "\n" in tok
        }

    def generate(self, user_prompt: str) -> FunctionCallResult:
        """Run the full constrained generation pipeline and
        return a structured function call."""
        prompt = self._build_prompt(user_prompt)
        context = self._encode(prompt)
        fn_name, context = self._generate_fn_name(context)
        fn_def = self.fn_map[fn_name]
        parameters, context = self._generate_parameters(
            context, fn_def, user_prompt
        )
        return FunctionCallResult(
            prompt=user_prompt, name=fn_name, parameters=parameters
        )

    def _generate_fn_name(self, context: list[int]) -> tuple[str, list[int]]:
        """Greedily decode a valid function name token by
        token using trie-based masking."""
        generated = ""
        current_context = list(context)
        for _ in range(80):
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
        self, logits: FloatArray, current: str
    ) -> FloatArray:
        """Allow only tokens that keep the
        generated text a valid trie prefix or complete word."""
        masked = np.full_like(logits, NEG_INF)
        for token_id, token_str in self.id_to_token.items():
            clean = self._clean(token_str)
            if not clean:
                continue
            candidate = current + clean
            if self.fn_trie.is_prefix(
                candidate
            ) or self.fn_trie.is_complete_word(candidate):
                masked[token_id] = logits[token_id]
        return masked if np.any(masked != NEG_INF) else logits.copy()

    def _best_fn_match(self, text: str) -> str:
        """Return the function name with the most characters
        in common with the given text."""
        best_name = self.fn_names[0]
        best_len = -1
        for name in self.fn_names:
            common = sum(1 for a, b in zip(text, name) if a == b)
            if common > best_len:
                best_len = common
                best_name = name
        return best_name

    def _generate_parameters(
        self, context: list[int], fn_def: FunctionDef, user_prompt: str = ""
    ) -> tuple[dict[str, Any], list[int]]:
        """Decode each parameter value according
        to its declared type in the function schema."""
        context = context + self._encode('", "parameters": {')
        params: dict[str, Any] = {}
        value: Any
        for i, (param_name, param_typedef) in enumerate(
            fn_def.parameters.items()
        ):
            separator = "" if i == 0 else ", "
            context += self._encode(f'{separator}"{param_name}": ')
            param_type = param_typedef.type
            if param_type in {ParamType.NUMBER, ParamType.INTEGER}:
                value, context = self._gen_number(context, param_type)
            elif param_type == ParamType.BOOLEAN:
                value, context = self._gen_boolean(context)
            else:
                context += self._encode('"')
                value, context = self._gen_string(context)
                context += self._encode('"')
            params[param_name] = value
        context += self._encode("}}")
        params = self._normalize_parameters(params, fn_def, user_prompt)
        return params, context

    def _normalize_parameters(
        self,
        params: dict[str, Any],
        fn_def: FunctionDef,
        user_prompt: str = "",
    ) -> dict[str, Any]:
        """Coerce parameter values to the types defined in the function schema."""
        params = params.copy()
        for param_name, param_typedef in fn_def.parameters.items():
            if param_name not in params:
                continue
            value = params[param_name]
            param_type = param_typedef.type
            if param_type == ParamType.INTEGER:
                if not isinstance(value, int) or isinstance(value, bool):
                    params[param_name] = int(
                        self._parse_number(str(value), ParamType.INTEGER)
                    )
            elif param_type == ParamType.NUMBER:
                if not isinstance(value, float):
                    params[param_name] = float(
                        self._parse_number(str(value), ParamType.NUMBER)
                    )
        return params

    def _gen_number(
        self, context: list[int], param_type: ParamType
    ) -> tuple[int | float, list[int]]:
        """Decode a numeric value token by token,
        stopping when the number is complete."""
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
            current_context.append(next_id)
            generated = candidate
            if self._is_complete_number(generated):
                next_logits = self._get_logits(current_context)
                next_id2 = int(np.argmax(next_logits))
                next_str = self._clean(self.id_to_token.get(next_id2, ""))
                if not self._would_extend_number(next_str, generated):
                    break
        return self._parse_number(generated, param_type), current_context

    def _mask_for_number(self, logits: FloatArray, current: str) -> FloatArray:
        """Allow only tokens that extend the current
        text into a valid numeric prefix."""
        masked = np.full_like(logits, NEG_INF)
        for token_id, token_str in self.id_to_token.items():
            text = self._clean(token_str)
            if (
                text
                and self._is_number_prefix(current + text)
                and len(current + text) <= 10
            ):
                masked[token_id] = logits[token_id]
        return masked if np.any(masked != NEG_INF) else logits.copy()

    def _is_number_prefix(self, text: str) -> bool:
        """Return True if text matches the pattern
        of an in-progress number literal."""
        return bool(re.match(r"^-?\d*\.?\d*$", text))

    def _is_complete_number(self, text: str) -> bool:
        """Return True if text can be fully parsed as a float."""
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _would_extend_number(self, next_str: str, current: str) -> bool:
        """Return True if appending next_str to current
        still forms a valid number prefix."""
        if not next_str:
            return False
        candidate = current + next_str
        return self._is_number_prefix(candidate) and len(candidate) <= 10

    def _parse_number(
        self, text: str, param_type: ParamType = ParamType.NUMBER
    ) -> int | float:
        """Parse text into an int or float, falling back to
        regex extraction on failure."""
        try:
            value = float(text)
        except ValueError:
            match = re.search(r"-?\d+\.?\d*", text)
            value = float(match.group()) if match else 0.0

        if param_type == ParamType.INTEGER:
            return int(value)
        return float(value)

    def _gen_boolean(self, context: list[int]) -> tuple[bool, list[int]]:
        """Select the highest-scoring token that cleanly
        maps to 'true' or 'false'."""
        logits = self._get_logits(context)
        true_score = NEG_INF
        false_score = NEG_INF
        true_id = None
        false_id = None
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

    def _is_string_continuation(self, token_id: int, token_raw: str) -> bool:
        """Return True if the token should be
        treated as part of an ongoing string value."""
        if token_id in self.stop_ids or token_id in self.newline_ids:
            return False
        decoded = token_raw.replace("Ġ", " ").replace("Ċ", "\n")
        stripped = decoded.lstrip()
        if not stripped:
            return True
        return stripped[0] not in ",}]"

    def _gen_string(self, context: list[int]) -> tuple[str, list[int]]:
        """Decode a string value token by token,
        stopping at a closing quote or newline."""
        parts: list[str] = []
        current_context = list(context)
        for _ in range(80):
            logits = self._get_logits(current_context)
            next_id = int(np.argmax(logits))
            token_raw = self.id_to_token.get(next_id, "")

            if next_id in self.stop_ids or next_id in self.newline_ids:
                break

            decoded = token_raw.replace("Ġ", " ").replace("Ċ", "\n")

            if decoded.strip() == '"':
                # Peek ahead to determine if the quote is embedded
                # or a string terminator
                look_context = current_context + [next_id]
                look_logits = self._get_logits(look_context)
                look_id = int(np.argmax(look_logits))
                look_raw = self.id_to_token.get(look_id, "")
                if self._is_string_continuation(look_id, look_raw):
                    parts.append('"')
                    current_context.append(next_id)
                    continue
                break

            if '"' in decoded:
                # Keep only the text before the closing quote
                before_quote = decoded.split('"', 1)[0]
                if before_quote:
                    parts.append(before_quote)
                break

            parts.append(decoded)
            current_context.append(next_id)

        return "".join(parts).strip(), current_context

    def _build_prompt(self, user_prompt: str) -> str:
        """Construct the system prompt that presents
        available functions and the user request."""
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
        """Strip whitespace markers
        and surrounding whitespace from a token."""
        return token_str.replace("Ġ", " ").replace("Ċ", "\n").strip()

    def _encode(self, text: str) -> list[int]:
        """Encode a string into a flat list of integer token
        IDs using the model's tokenizer."""
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

    def _get_logits(self, token_ids: list[int]) -> FloatArray:
        """Run a forward pass and return a flat float64 logits array
        for the next token."""
        raw = self.model.get_logits_from_input_ids(token_ids)
        while (
            isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list)
        ):
            raw = raw[0]
        return np.array(raw, dtype=np.float64).flatten()
