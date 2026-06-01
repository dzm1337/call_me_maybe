# call_me_maybe

**Function Calling with Constrained Decoding for Small LLMs**

A lightweight, efficient framework for turning natural language into precise tool/function calls using constrained decoding — no fine-tuning required.

---

## Features

- **True constrained decoding**: Forces the model to output valid function names and correctly formatted parameters
- **Zero-shot function calling**: Works with small local models
- **Type-aware generation**: Handles strings, integers, floats, and booleans with proper validation
- **Trie-based function name enforcement**: Guarantees the model picks from your defined functions
- **Lightweight & fast**: Built for local inference with small models
- **Clean architecture**: Modular design with Pydantic models and clean separation of concerns

---

## How It Works

Instead of relying on the LLM to "hopefully" output valid JSON, `call_me_maybe` uses **logit masking** during generation to:

1. Restrict function name tokens to only those in your function definitions
2. Enforce correct parameter structure and types
3. Guide the model toward valid outputs using a Trie for function names

This dramatically improves reliability compared to standard prompting.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/dzm1337/call_me_maybe.git
cd call_me_maybe

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

Quick Start
1. Define your functions
Create data/input/functions_definition.json:
JSON[
  {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
      "city": {"type": "string"},
      "units": {"type": "string"}
    }
  },
  {
    "name": "search_web",
    "description": "Search the internet for information",
    "parameters": {
      "query": {"type": "string"},
      "num_results": {"type": "integer"}
    }
  }
]
2. Run the tool
Bashuv run python -m src
Or with custom paths:
Bashuv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/results.json

Project Structure
textcall_me_maybe/
├── src/
│   ├── __main__.py                 # CLI entry point
│   ├── constrained_decoder.py      # Core constrained decoding logic
│   ├── models.py                   # Pydantic schemas
│   ├── vocabulary.py               # Tokenizer utilities
│   └── loader.py                   # Input loading
├── llm_sdk/                        # Local LLM wrapper (custom)
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
├── pyproject.toml
└── Makefile

Technical Details
Constrained Decoding Strategy

Function names: Trie-based token masking to ensure exact match
Parameters: Structured generation with type-specific generators
String handling: Quote-aware generation with continuation logic
Numbers: Regex-validated numeric token filtering

Supported Parameter Types

string
number
integer
boolean


Development
Bash# Install dev dependencies
uv sync

# Run the project
make run

# Lint & type check
make lint

# Strict checks
make lint-strict

Dependencies

Core: pydantic, numpy
LLM: Custom llm-sdk (local transformers-based inference)
Dev: flake8, mypy, uv


Use Cases

AI Agents — Reliable tool use
Local automation — Voice → Function calling
RAG systems — Structured query routing
Edge AI — Function calling on small models
Research — Experimenting with constrained generation


Roadmap

 Support for optional parameters
 Array and object type support
 Streaming output
 Multiple function calls (parallel)
 Better error recovery and fallback strategies
 Evaluation metrics and test harness
