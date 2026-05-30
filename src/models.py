from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParamType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    INTEGER = "integer"


class TypeDef(BaseModel):
    type: ParamType


class FunctionDef(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(...)
    parameters: dict[str, TypeDef] = Field(...)
    returns: TypeDef | None = None


class FunctionCallResult(BaseModel):
    prompt: str = Field(...)
    name: str = Field(...)
    parameters: dict[str, Any] = Field(...)
    model_config = ConfigDict(extra="forbid")
