from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParamType(str, Enum):
    """Supported parameter types in function definitions."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class TypeDef(BaseModel):
    """Type wrapper for a parameter or return value."""

    type: ParamType


class FunctionDef(BaseModel):
    """Complete definition of a callable function."""

    name: str = Field(..., min_length=1)
    description: str = Field(...)
    parameters: dict[str, TypeDef] = Field(...)
    returns: TypeDef | None = None


class FunctionCallResult(BaseModel):
    """Structured result of a function call prediction."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(...)
    name: str = Field(...)
    parameters: dict[str, Any] = Field(...)

