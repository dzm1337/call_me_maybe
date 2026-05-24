from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ParameterType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class TypeDef(BaseModel):
    type: ParameterType


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, TypeDef]
    returns: TypeDef | None


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]

    model_config = ConfigDict(extra="forbid")
