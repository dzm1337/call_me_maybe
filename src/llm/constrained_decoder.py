import json
from src.llm.client import LLMClient
from src.models.function import FunctionDef


class ConstrainedDecoder:
    def __init__(self, llm_client: LLMClient, functions: list[FunctionDef]):
        self.llm = llm_client
        self.functions = functions
        self.func_names = [x.name for x in functions]
        self.map_func = {x.name: x for x in functions}
        self.vocab = self.llm._load_vocabulary() # {id: text}
        self.text_to_token = {text: tid for tid, text in self.vocab.items()}

    def _fabricate_prompt(self) -> str:
        """Fabricating a prompt with allowed functions"""
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

    def _full_prompt(self, user_prompt: str) -> str:
        fabricated_prompt = self._fabricate_prompt()
        return f"{fabriacted_prompt}\nUser: {user_prompt}\nAssistant: "

    