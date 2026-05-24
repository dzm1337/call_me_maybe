import json
from pathlib import Path

from src.models.function import FunctionDef


def load_file(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_functions() -> list[FunctionDef]:
    file_path = (
        Path(__file__).parent.parent
        / "data"
        / "input"
        / "functions_definition.json"
    )

    data = load_file(file_path)

    if not isinstance(data, list):
        raise ValueError("functions_definition.json deve conter uma lista")

    functions: list[FunctionDef] = []
    for item in data:
        fn = FunctionDef.model_validate(item)
        functions.append(fn)

    return functions


def load_prompts() -> list[str]:
    file_path = (
        Path(__file__).parent.parent
        / "data"
        / "input"
        / "function_calling_tests.json"
    )

    data = load_file(file_path)

    if not isinstance(data, list):
        raise ValueError("function_calling_tests.json deve conter uma lista")

    prompts: list[str] = []
    for item in data:
        if not isinstance(item, dict) or "prompt" not in item:
            raise ValueError(f"Item inválido: {item}")
        prompts.append(item["prompt"])

    return prompts
