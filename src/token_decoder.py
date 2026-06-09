import re
import numpy as np
import numpy.typing as npt
from typing import TypeAlias
from src.models import ParamType
from llm_sdk import Small_LLM_Model  # type: ignore

NEG_INF: float = float("-inf")
FloatArray: TypeAlias = npt.NDArray[np.float64]


class TokenDecoder:
    """
    Handles primitive value decoding under constrained token masking.

    This class enforces schema-level validity
    without interfering with higher-level JSON structure.
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        id_to_token: dict[int, str],
        stop_ids: set[int],
        newline_ids: set[int],
    ) -> None:
        """
        Initialize decoder with vocabulary and structural token filters.

        stop_ids and newline_ids prevent malformed JSON generation.
        """
        self.model = model
        self.id_to_token = id_to_token
        self.stop_ids = stop_ids
        self.newline_ids = newline_ids

    def _get_logits(self, context: list[int]) -> FloatArray:
        """
        Retrieve next-token logits from the language model.

        Ensures output is flattened into a 1D numpy array.
        """
        raw = self.model.get_logits_from_input_ids(context)

        while isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]

        return np.array(raw, dtype=np.float64).flatten()

    def _clean(self, token_str: str) -> str:
        """
        Normalize tokenizer markers.

        Removes subword spacing symbols while preserving content.
        """
        return token_str.replace("Ġ", " ").replace("Ċ", "\n")

    def _mask_for_boolean(self, logits: FloatArray) -> FloatArray:
        """
        Restrict logits so that only tokens representing the literals
        "true" or "false" remain selectable. All other tokens are masked
        to negative infinity to enforce schema-compliant boolean output.
        """
        masked = np.full_like(logits, NEG_INF)

        for tid, token_str in self.id_to_token.items():
            clean = self._clean(token_str).strip().lower()

            if clean in {"true", "false"}:
                masked[tid] = logits[tid]

        return masked if np.any(masked != NEG_INF) else logits.copy()

    def gen_boolean(self, context: list[int]) -> tuple[bool, list[int]]:
        """
        Generate a boolean value using constrained decoding.

        Only tokens matching 'true' or 'false' are permitted.
        """
        logits = self._get_logits(context)
        masked = self._mask_for_boolean(logits)

        next_id = int(np.argmax(masked))
        token_str = self._clean(self.id_to_token[next_id]).strip().lower()

        if token_str == "true":
            value = True
        elif token_str == "false":
            value = False
        else:
            raise ValueError(f"Invalid boolean token generated: {token_str}")

        return value, context + [next_id]

    def _mask_for_string(
        self,
        logits: FloatArray,
        current: str,
    ) -> FloatArray:
        """
        Restrict tokens that would break JSON validity or
        produce malformed string values.
        """
        masked = np.full_like(logits, NEG_INF)

        for tid, token_str in self.id_to_token.items():
            if tid in self.stop_ids or tid in self.newline_ids:
                continue

            decoded = self._clean(token_str)

            # Prevent raw newline inside JSON strings.
            if "\n" in decoded:
                continue

            # Prevent multiple backslashes in a single token.
            if decoded.count("\\") > 1:
                continue

            candidate = current + decoded

            # Prevent sequences of backslashes.
            if re.search(r"\\\\{3,}", candidate):
                continue

            masked[tid] = logits[tid]

        return masked if np.any(masked != NEG_INF) else logits.copy()

    def _is_string_continuation(self, token_id: int, token_raw: str) -> bool:
        """Return True if the token should be
        treated as part of an ongoing string value."""
        if token_id in self.stop_ids or token_id in self.newline_ids:
            return False
        decoded = self._clean(token_raw)
        stripped = decoded.lstrip()
        if not stripped:
            return True
        return stripped[0] not in ",}]"

    def gen_string(self, context: list[int]) -> tuple[str, list[int]]:
        parts: list[str] = []
        current_context = list(context)
        generated = ""

        for _ in range(80):
            logits = self._get_logits(current_context)
            masked = self._mask_for_string(logits, generated)

            next_id = int(np.argmax(masked))
            token_raw = self.id_to_token.get(next_id, "")
            decoded = self._clean(token_raw)

            # If token contains a quote anywhere,
            # treat everything after the first quote as structural.
            if '"' in decoded:
                before_quote = decoded.split('"', 1)[0]
                if before_quote:
                    parts.append(before_quote)
                    generated += before_quote
                break

            parts.append(decoded)
            generated += decoded
            current_context.append(next_id)

        return "".join(parts).lstrip(), current_context

    def _is_number_prefix(self, text: str) -> bool:
        """Return True if text matches a partial numeric pattern."""
        return bool(re.match(r"^-?\d*\.?\d*$", text))

    def _is_complete_number(self, text: str) -> bool:
        """Return True if text can be parsed as float."""
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _would_extend_number(self, next_str: str, current: str) -> bool:
        """Return True if appending next_str to current_context
        still forms a valid number prefix."""
        if not next_str:
            return False
        candidate = current + next_str
        return self._is_number_prefix(candidate) and len(candidate) <= 10

    def _mask_for_number(
        self,
        logits: FloatArray,
        current: str,
    ) -> FloatArray:
        """Mask tokens that would break numeric validity."""
        masked = np.full_like(logits, NEG_INF)

        for tid, token_str in self.id_to_token.items():
            text = self._clean(token_str)

            if text and self._is_number_prefix(current + text):
                masked[tid] = logits[tid]

        return masked if np.any(masked != NEG_INF) else logits.copy()

    def _parse_number(
        self,
        text: str,
        param_type: ParamType,
    ) -> int | float:
        """Convert numeric text to correct Python type."""
        try:
            value = float(text)
        except ValueError:
            match = re.search(r"-?\d+\.?\d*", text)
            value = float(match.group()) if match else 0.0

        if param_type == ParamType.INTEGER:
            return int(value)

        return float(value)

    def gen_number(
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
