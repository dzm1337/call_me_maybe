from llm_sdk.llm_sdk import Small_LLM_Model


class LLMClient(Small_LLM_Model):
    def __init__(self):
        Small_LLM_Model.__init__(self)

    def get_vocac(self):
        return Small_LLM_Model.get_path_to_vocab_file()
