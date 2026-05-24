from src.llm.client import LLMClient

from src.validation.loader import load_prompts, validate_functions


def main():
    print("Loading files...\n")

    # Load functions and prompts
    functions = validate_functions()
    prompts = load_prompts()

    print(f"✅ Loaded {len(functions)} functions")
    print(f"✅ Loaded {len(prompts)} prompts")

    # Initialize LLM client
    llm = LLMClient()
    print("\n🤖 LLM Client ready\n")

    # Encode ALL prompts
    print("=" * 60)
    print("ENCODING ALL PROMPTS")
    print("=" * 60)

    for i, prompt in enumerate(prompts, 1):
        token_ids = llm.encode(prompt)
        print(f"\n[{i}] Prompt: {prompt}")
        print(f"    Tokens: {token_ids}")
        print(f"    Token count: {len(token_ids)}")

    # Encode ALL function definitions as JSON strings
    print("\n" + "=" * 60)
    print("ENCODING ALL FUNCTION DEFINITIONS")
    print("=" * 60)

    for i, fn in enumerate(functions, 1):
        fn_json = fn.model_dump_json()
        token_ids = llm.encode(fn_json)
        print(f"\n[{i}] Function: {fn.name}")
        print(f"    JSON: {fn_json[:100]}...")
        print(f"    Token count: {len(token_ids)}")

    # Encode everything combined
    print("\n" + "=" * 60)
    print("ENCODING EVERYTHING TOGETHER")
    print("=" * 60)

    all_text = "System: You are a function calling assistant.\n\n"
    all_text += "Available functions:\n"
    for fn in functions:
        all_text += f"- {fn.name}: {fn.description}\n"
    all_text += "\nUser prompts:\n"
    for prompt in prompts[:3]:
        all_text += f"- {prompt}\n"

    token_ids = llm.encode(all_text)
    print(f"\nTotal text length: {len(all_text)} characters")
    print(f"Total tokens: {len(token_ids)}")
    print(f"\nFirst 20 tokens: {token_ids[:20]}")


if __name__ == "__main__":
    main()
