import argparse
import json
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.constrained_decoder import ConstrainedDecoder
from src.loader import load_functions, load_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Function Calling with Constrained Decoding"
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the JSON file defining available functions.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the JSON file containing prompts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path where the output JSON file will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Load inputs, run constrained decoding, and write results."""
    args = parse_args()

    print("Loading functions and prompts...")
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"{len(functions)} function(s) loaded.")
    print(f"{len(prompts)} prompt(s) loaded.")

    print("Initializing LLM model...")
    try:
        model = Small_LLM_Model()
    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        sys.exit(1)

    decoder = ConstrainedDecoder(model, functions)
    results = []
    total = len(prompts)
    print(f"\nProcessing {total} prompt(s)...\n")

    for i, prompt_text in enumerate(prompts, 1):
        display = (
            prompt_text[:60] + "..." if len(prompt_text) > 60 else prompt_text
        )
        print(f"[{i:2d}/{total}] {display}")
        try:
            result = decoder.generate(prompt_text)
            results.append(result.model_dump())
            print(f"{result.name}({result.parameters})")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append(
                {
                    "prompt": prompt_text,
                    "name": "error",
                    "parameters": {},
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to: {args.output}")
    except OSError as e:
        print(f"ERROR: Could not write output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
