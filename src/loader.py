import json
from json import JSONDecodeError
from pathlib import Path
from src.function import FunctionDef
from pydantic import ValidationError

def validate_functions() -> list[FunctionDef]:
    file_path = (
        Path(__file__).parent.parent
        / "data"
        / "input"
        / "functions_definition.json"
    )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load functions definition: {e}")

    if not isinstance(data, list):
        raise ValueError("functions_definition.json must be a list")

    functions: list[FunctionDef] = []
    try:
        for item in data:
            fn = FunctionDef.model_validate(item)
            functions.append(fn)
    except ValidationError as e:
        raise ValueError(f"Invalid function definition: {e}")
    return functions


def load_prompts() -> list[str]:
    file_path = (
        Path(__file__).parent.parent
        / "data"
        / "input"
        / "function_calling_tests.json"
    )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load prompts: {e}")

    if not isinstance(data, list):
        raise ValueError("function_calling_tests.json must be a list")

    prompts: list[str] = []
    for item in data:
        if not isinstance(item, dict) or "prompt" not in item:
            raise ValueError(f"Invalid Item: {item}")
        prompts.append(item["prompt"])

    return prompts
