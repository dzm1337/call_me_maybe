from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParamType(str, Enum):
    """Enumeration of supported primitive parameter types."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    INTEGER = "integer"


class TypeDef(BaseModel):
    """Schema wrapper describing the type of a single parameter."""

    type: ParamType


class FunctionDef(BaseModel):
    """Definition of a callable function and its typed parameters."""

    name: str = Field(..., min_length=1)
    description: str = Field(...)
    parameters: dict[str, TypeDef] = Field(...)
    returns: TypeDef | None = None


class FunctionCallResult(BaseModel):
    """Structured result representing a generated function call."""

    prompt: str = Field(...)
    name: str = Field(...)
    parameters: dict[str, Any] = Field(...)
    model_config = ConfigDict(extra="forbid")
