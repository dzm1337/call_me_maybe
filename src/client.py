from llm_sdk.llm_sdk import Small_LLM_Model
import json
from json import JSONDecodeError
from typing import Dict, List


class LLMClient(Small_LLM_Model):
    def __init__(self):
        super().__init__()
        self._vocab = None # token_id -> token_text
        self._text_to_token = None # token_text -> token_id

    def _load_vocabulary(self) -> Dict[int, str]:
        """Loading the vocabulary manually"""
        if self._vocab is None:
            try:

                path = self.get_path_to_vocab_file()
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._vocab = {int(t_id): t_text for t_id, t_text in raw.items()}
                """Inversion for fast lookup"""
                self._text_to_token = {t_text: t_id for t_id, t_text in self._vocab.items()}

            except (FileNotFoundError, PermissionError, JSONDecodeError) as e:
                raise RuntimeError(f"ERROR: Failed to open {path}: {e}")
        return self._vocab
    
    def text_to_tokens(self, text: str) -> List[int]:
        """algorithm to find the longest substring that we have inside our vocabulary
        from the current position"""
        self._load_vocabulary()
        tokens = []
        i = 0
        n = len(text)

        while i < n:
            matched = False
            max_len = n - i if n - i < 20 else 20
            for length in range(max_len, 0, -1):
                possible_token = text[i:i+length]

                if possible_token in self._text_to_token:
                    tokens.append(self._text_to_token[possible_token])
                    i += length
                    matched = True
                    break

            if not matched:
                first_char = text[i]

                if first_char in self._text_to_token:
                    tokens.append(self._text_to_token[first_char])
                else:
                    tokens.append(0)
                    
                i += 1

        return tokens

    def tokens_to_text(self, tokens_id: list[int]) -> str:
        self._load_vocabulary()
        return "".join(self._vocab.get(tid, "") for tid in tokens_id)

    def get_logits(self, token_ids: list[int]):
        return self.get_logits_from_input_ids([token_ids])