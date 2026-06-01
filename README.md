*This project has been created as part of the 42 curriculum by dde-paul.*

# call_me_maybe

**Introduction to Function Calling in LLMs using Constrained Decoding**

A robust implementation that enables small language models (Qwen3-0.6B) to reliably translate natural language into structured, schema-compliant function calls.

---

## Description

This project demonstrates how to bridge the gap between natural language and executable code by implementing **constrained decoding** for function calling.

Instead of relying on fragile prompting, the system forces the LLM to generate **100% valid JSON** that strictly follows the provided function definitions (name + typed parameters). This approach dramatically improves reliability on small models.

**Goal**: Achieve near-perfect function selection and argument extraction while guaranteeing syntactically and semantically valid output.

---

## Features

- True token-level constrained decoding using logit masking
- Trie-based function name enforcement
- Type-aware parameter generation (string, number, integer, boolean)
- Full Pydantic validation
- Clean separation between prompt handling, decoding logic, and LLM interaction
- Robust error handling and graceful failure modes

---

## Instructions

### Installation

```bash
# Create virtual environment and install dependencies
uv sync
Usage
Bash# Default execution (uses data/input/ paths)
uv run python -m src

# With custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

Makefile Commands
make run        # Run the program
make debug      # Run with Python debugger
make lint       # Run flake8 + mypy
make lint-strict # Strict type checking
make clean      # Clean cache files

Algorithm Explanation (Constrained Decoding)
The core innovation is constrained decoding:

Function Name Selection: A Trie is built from all available function names. At each generation step, only tokens that can lead to a valid function name are allowed (others are masked with -inf in the logits).
JSON Structure Enforcement: After the function name, the decoder enforces the exact JSON schema:
Opening {
Parameter keys (only those defined in the function)
Type-specific value generation

Type-Aware Generation:
Strings: Quote-aware generation with proper escaping
Numbers: Regex-validated numeric tokens
Booleans: Restricted to true/false

LLM Interaction: Uses only the allowed llm_sdk interface (get_logits_from_input_ids, vocabulary mapping, etc.) without accessing private methods.

This guarantees that every generated output is always valid JSON and matches the function schema.

Design Decisions

Pydantic Models: All data structures (FunctionDefinition, FunctionCall, etc.) are validated with Pydantic.
Modular Architecture: constrained_decoder.py handles all masking logic, keeping the main pipeline clean.
No Heuristics: Function selection is done purely by the LLM through constrained generation.
Vocabulary-Driven: Heavy use of the vocabulary JSON to map tokens → strings for validation.
Error Resilience: Comprehensive try/except blocks and clear user messages.


Performance Analysis

Accuracy: >90% correct function + parameter extraction on test cases
Reliability: 100% valid JSON output (no parsing errors)
Speed: Processes all provided test cases in under 5 minutes on standard hardware
Model: Qwen/Qwen3-0.6B

Constrained decoding proves far more effective than raw prompting for small models.

Challenges Faced & Solutions

Complex Tokenization: Solved by working directly with vocabulary mappings and careful string continuation logic.
String Generation: Implemented quote detection and multi-token string continuation.
Number Parsing: Built custom numeric token filtering using regex patterns.
SDK Restrictions: Strictly respected public API only.
Edge Cases: Added extensive testing for empty strings, special characters, and malformed inputs.


Testing Strategy

Manual testing with varied prompts and function definitions
Edge case coverage (empty inputs, invalid JSON, missing files, ambiguous prompts)
Validation of output against schema using Pydantic
Repeated runs to ensure deterministic reliability under constraints


Resources
Learning Resources

Structured Output from LLMs: Grammars, Regex, and State Machines
https://www.youtube.com/watch?v=xpvFinvqRCA&t=389s 

A Guide to Structured Generation Using Constrained Decoding
https://www.aidancooper.co.uk/constrained-decoding/

Part 6: Implementing Constrained Decoding
https://medium.com/@albersj66/part-6-implementing-constrained-decoding-for-phi-3-vision-2c72a1be6a17

Coalescence: making LLM inference 5x faster
https://blog.dottxt.ai/coalescence.html



AI Usage
Help structure the README.md
Brainstorm constrained decoding strategies
Generate initial test cases

Example Usage
Input prompt: "What is the sum of 40 and 2?"
Output:
JSON{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40.0,
    "b": 2.0
  }
}
