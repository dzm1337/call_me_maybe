from enum import Enum

from pydantic import BaseModel, Field


class ParameterType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class ParameterDef(BaseModel):
    type: ParameterType


class FunctionDef(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, ParameterDef] = Field(...)
    returns: ParameterDef | None
