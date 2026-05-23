from src.validation.loader import load_prompts, validate_functions

try:
    prompts = load_prompts()
    validated_functions = validate_functions()
    for item in prompts:
        print(f"{item} | ")
    for item in validated_functions:
        print(f"{item} | ")
except ValueError as e:
    print(e)
