import json
from pathlib import Path

from src.models.function import FunctionDef


def load_file(file_path: Path):
    with open(file_path, "r") as f:
        json_file = json.load(f)
    return json_file


def validate_functions() -> list[FunctionDef]:
    json_file = load_file(
        Path(__file__).parent.parent
        / "data"
        / "input"
        / "functions_definition.json"
    )
    functions: list[FunctionDef] = []
    for item in json_file:
        fn = FunctionDef.model_validate(item)
        functions.append(fn)
    return functions


def load_prompts() -> list[str]:
    json_file = load_file(
        Path(__file__).parent.parent
        / "data"
        / "input"
        / "function_calling_tests.json"
    )
    if not isinstance(json_file, list):
        raise ValueError("ERROR: Invalid type")
    prompts: list[str] = []
    for item in json_file:
        if not isinstance(item, dict) or "prompt" not in item:
            raise ValueError(f"ERROR: Invalid item {item}")
        prompts.append(item["prompt"])

    return prompts
