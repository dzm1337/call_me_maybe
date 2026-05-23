from pydantic import BaseModel, ConfigDict


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
    model_config = ConfigDict(extra="forbid")
