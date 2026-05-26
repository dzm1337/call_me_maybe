import json
from pathlib import Path

from pydantic import ValidationError

from src.models import FunctionDef


def load_functions(path: Path) -> list[FunctionDef]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Functions file not found: {path}")
    except PermissionError:
        raise RuntimeError(f"Permission denied reading: {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in functions file {path}: {e}")

    if not isinstance(data, list):
        raise ValueError(
            f"Functions file must contain a JSON array, got: {type(data).__name__}"
        )

    functions: list[FunctionDef] = []
    for i, item in enumerate(data):
        try:
            functions.append(FunctionDef.model_validate(item))
        except ValidationError as e:
            raise ValueError(f"Invalid function definition at index {i}: {e}")

    return functions


def load_prompts(path: Path) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Prompts file not found: {path}")
    except PermissionError:
        raise RuntimeError(f"Permission denied reading: {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in prompts file {path}: {e}")

    if not isinstance(data, list):
        raise ValueError(
            f"Prompts file must contain a JSON array, got: {type(data).__name__}"
        )

    prompts: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "prompt" not in item:
            raise ValueError(
                f"Item at index {i} must be a dict with a 'prompt' key, "
                f"got: {item}"
            )
        prompts.append(str(item["prompt"]))

    return prompts
