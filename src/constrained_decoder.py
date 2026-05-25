import json
from src.client import LLMClient
from src.models import FunctionDef, FunctionCallResult
from enum import Enum
import numpy as np
from typing import Set 

class State(str, Enum):
    START = "START"
    IN_OBJECT = "IN_OBJECT"
    IN_KEY = "IN_KEY"
    AFTER_KEY = "AFTER_KEY"
    IN_STRING = "IN_STRING"
    IN_NUMBER = "IN_NUMBER"
    EXPECT_COMMA_OR_CLOSE = "EXPECT_COMMA_OR_CLOSE"
    DONE = "DONE"

class ConstrainedDecoder:
    def __init__(self, llm_client: LLMClient, functions: list[FunctionDef]):
        self.llm = llm_client
        self.functions = functions
        self.func_names = [x.name for x in functions] # all the name of the functions in json file
        self.map_func = {x.name: x for x in functions} #
        self.vocab = self.llm._load_vocabulary() # {id: text}
        self.text_to_token = {text: tid for tid, text in self.vocab.items()} # {text: id}

    def _fabricate_prompt(self) -> str:
        """Fabricating a prompt with allowed functions from the json file"""
        lines = ["Available Functions:"]
        for f in self.functions:
            params = ", ".join(f"{pname}: {ptype.type.value}" for pname, ptype in f.parameters.items())
            if f.returns:
                ret = f.returns.type.value
            else:
                ret = "void"
            lines.append(f"{f.name}({params}) -> {ret}")
        lines.append('Output format: {"prompt": "...", "name": "...", "parameters": {...}}')
        lines.append("Respond with JSON only.")
        return "\n".join(lines)

    def _generate_full_prompt(self, user_prompt: str) -> str:
        fabricated_prompt = self._fabricate_prompt()
        return f"{fabricated_prompt}\nUser: {user_prompt}\nAssistant: "

    def _apply_mask(self, logits: np.ndarray, allowed_ids: Set[int]) -> np.darray:
        """set all tokens that are not allowed to -inf (absolute infinite) and check the allowed ones 
        and put its original logit"""
        masked = np.full_like(logits, -np.inf)
        for t_id in allowed_ids:
            if t_id < len(masked):
                masked[t_id] = logits[t_id]
        return masked

    def _init_state(self) -> dict:
        """INITIAL FSM state: before any JSON token"""
        return {
            "phase": State.START,
            "current_key": "",
            "buffer": ""
            "inside_params": False,
        }

    def _allowed_tokens(self, state: dict) -> Set[int]:
        """
        Because of the current FSM state, return the set of token IDs that can be used
        reinforcing the JSON strucutre and schema constraints
        """
        phase = phase["state"]
        allowed = set()

        def add_char(ch: str):
            if ch in self.text_to_token:
                allowed.add(self.text_to_token[ch])

        if phase == State.START:
            add.char('{')

        elif phase == State.IN_OBJECT:
            add.char('"')

        elif phase == State.IN_KEY:
            """reading the key name, any token is allowed as long it doesn't contain a ""
            because "" would end the key" ex: "prompt" """
            for t_id, t_text in self.vocab:
                if '"' not in t_text:
                    allowed.add(t_id)

        elif phase == State.AFTER_KEY:
            add_char(':') # after a key in JSON we always have a ':'

        elif phase == State.IN_STRING:
            for t_id, t_text in self.vocab.items():
                if '"' not in t_text:
                    allowed.add(t_id)

        elif phase == State.IN_NUMBER:
            for ch in "0123456789.-":
                if ch in self.text_to_token:
                    allowed.add(text_to_token[ch])

        elif phase == State.EXPECT_COMMA_OR_CLOSE:
            add_char(',')
            add_char('}')
        
        else:
            #fallback
            allowed.update(self.vocab.keys())
        
        if state["current_key"] == name and phase == State.IN_STRING:
        
        return allowed
