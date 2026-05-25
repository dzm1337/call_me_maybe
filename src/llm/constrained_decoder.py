import json
from src.llm.client import LLMClient
from src.models.function import FunctionDef


class ConstrainedDecoder:
    def __init__(self, llm_client: LLMClient, functions: list[FunctionDef]):
        self.llm = llm_client
        self.functions = functions
        self.func_names = [x.name for x in functions]
        self.map_func = {x.name: x for x in functions}
        self.vocab = self._load_vocabulary()
        self.text_to_token = self._text_to_token()

    def _text_to_token(self) -> Dict[str, int]:
        """Inverting the dictionary to token_str: token_id"""
        vocab = self.vocab
        mapping = {}
        for token_id, token_str in vocab.items():
            if token_str not in mapping:
                mapping[token_str] = token_id
        return mapping 

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

    def generate(self, user_prompt: str) -> FunctionCallResult:
        fabricated_prompt = self._fabricate_prompt()
        full_prompt = f"{fabriacted_prompt}\nUser: {user_prompt}\nAssistant: "

        input_ids = self.llm.encode(full_prompt)