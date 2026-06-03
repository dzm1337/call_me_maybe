import json

from llm_sdk import Small_LLM_Model  # type: ignore


def load_vocabulary(
    model: Small_LLM_Model,
) -> tuple[dict[int, str], dict[str, int]]:
    """Load vocabulary JSON and build ID→token and token→ID mappings."""
    try:
        vocab_path = model.get_path_to_vocab_file()
    except Exception as e:
        raise RuntimeError(f"Failed to get vocabulary path: {e}")

    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw: dict[str, int] = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Vocabulary file not found: {vocab_path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in vocabulary file: {e}")

    id_to_token: dict[int, str] = {}
    token_to_id: dict[str, int] = {}

    # Convert raw string→id mapping into validated integer-based dictionaries
    for token_str, token_id in raw.items():
        try:
            tid = int(token_id)
            id_to_token[tid] = token_str
            token_to_id[token_str] = tid
        except (ValueError, TypeError):
            continue

    return id_to_token, token_to_id
