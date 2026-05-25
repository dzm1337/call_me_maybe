from pathlib import Path
from src.validation.loader import load_prompts, validate_functions
import json


def main():
    def fabricating_prompt(functions) -> str:
        lines = ["Available Functions:"]
        for f in functions:
            params = ", ".join(f"{pname}: {ptype.type.value}" for pname, ptype in f.parameters.items())
            if f.returns:
                ret = f.returns.type.value
            else:
                ret = "void"
            lines.append(f"{f.name}({params}) -> {ret}")
        lines.append('Output format: {"prompt": "...", "name": "...", "parameters": {...}}')
        lines.append("Respond with JSON only.")
        return "\n".join(lines)
    x = validate_functions()
    print(fabricating_prompt(x))
if __name__ == "__main__":
    main()
