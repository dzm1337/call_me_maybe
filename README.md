  # call_me_maybe

  **Introduction to Function Calling in LLMs using Constrained Decoding**

  *A robust implementation that enables small language models (Qwen3-0.6B) to reliably translate natural language into structured, schema-compliant function calls.*

  ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
  ![Qwen](https://img.shields.io/badge/Model-Qwen3--0.6B-blue)

  ---

  ##  Table of Contents
  - [About the Project](#about-the-project)
  - [Key Features](#key-features)
  - [Why Constrained Decoding?](#why-constrained-decoding)
  - [How It Works](#how-it-works)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Examples](#examples)
  - [Performance](#performance)
  - [Testing Strategy](#testing-strategy)
  - [Design Decisions](#design-decisions)
  - [Resources](#resources)
  - [AI Assistance](#ai-assistance)
  - [License](#license)

  ---

  ## About the Project

  This project was created as part of the **42 curriculum** by **dde-paul**.

  `call_me_maybe` demonstrates a production-grade approach to **function calling** with small language models. Instead of relying on brittle prompt engineering, it uses **token-level constrained decoding** to *force* the model to generate **100% valid JSON** that strictly adheres to a provided function schema.

  By combining a Trie-based function name selector with type-aware JSON generation and logit masking, the system achieves dramatically higher reliability than traditional prompting methods — especially on models as small as **Qwen3-0.6B**.

  **Goal**: Near-perfect function selection and argument extraction with **guaranteed syntactic and semantic validity**.

  ---

  ## Key Features

  - **True constrained decoding** via logit masking at every generation step
  - **Trie-based function name enforcement** for zero ambiguity
  - **Type-aware generation** (string, number, integer, boolean)
  - **Full Pydantic validation** on both input definitions and output
  - **Clean modular architecture** with separation of concerns
  - **Robust error handling** and graceful degradation
  - **Vocabulary-driven token filtering** using the model's official vocabulary
  - **No private LLM SDK access** — respects public interface only

  ---

  ## Why Constrained Decoding?

  Traditional function calling methods depend heavily on the model "following instructions." Small models frequently:
  - Hallucinate function names
  - Produce invalid JSON
  - Use wrong parameter types
  - Miss required fields

  **Constrained decoding** solves this by **mathematically eliminating invalid paths** during generation. The model is only ever allowed to produce tokens that lead to valid output.

  ---

  ## How It Works

  ### Core Algorithm

  1. **Function Name Selection**
    - A Trie is built from all available function names
    - At each decoding step, only tokens that continue a valid function name are allowed

  2. **JSON Structure Enforcement**
    - After the function name, the decoder forces the exact JSON schema:
      - Opening brace
      - Only allowed parameter keys
      - Correct type per parameter
      - Proper closing

  3. **Type-Aware Token Generation**
    - **Strings**: Quote detection + escaping + multi-token continuation
    - **Numbers**: Regex-validated numeric tokens
    - **Booleans**: Restricted to `True`/`False`
    - **Integers**: Additional constraints on decimal points

  4. **Final Validation**
    - Every output is parsed and validated using Pydantic models

  ---

  ## Project Structure

  ```bash
  call_me_maybe/
  ├── src/
  │   ├── __main__.py
  │   ├── constrained_decoder.py     # Core constrained decoding logic
  │   ├── models.py                  # Pydantic data models
  │   ├── utils.py
  │   └── llm/                       # LLM interaction layer
  ├── data/
  │   ├── input/
  │   │   ├── functions_definition.json
  │   │   └── function_calling_tests.json
  │   └── output/
  ├── tests/
  ├── pyproject.toml
  ├── Makefile
  └── README.md

  Installation
  Bash
  # Create virtual environment and install dependencies
  uv sync
  Usage
  Basic Usage
  Bash
  uv run python -m src
  With Custom Paths
  Bash
  uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
  Makefile Commands
  Bash
  make run          # Run the program
  make debug        # Run with pdb
  make lint         # Run flake8 + mypy
  make lint-strict  # Strict type checking
  make clean        # Clean cache files

  Examples
  Input
  JSON
  {
    "prompt": "What is the sum of 40 and 2?"
  }
  Output
  JSON
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 40.0,
      "b": 2.0
    }
  }

  Performance
  Accuracy: >90% correct function + parameter extraction

  Reliability: 100% valid JSON output (no parsing failures)

  Speed: Processes all test cases in under 5 minutes on standard hardware

  Model: Qwen/Qwen3-0.6B (small but highly effective with constraints)

  Testing Strategy
  Comprehensive manual testing with diverse natural language prompts

  Edge case coverage (empty inputs, special characters, ambiguous requests)

  Schema validation using Pydantic

  Repeated runs to verify deterministic behavior under constraints

  Design Decisions
  Pydantic everywhere for strong data validation

  No heuristics — pure constrained generation

  Modular decoder — easy to extend to new types or grammars

  Educational focus — code is heavily commented and structured for learning

  Resources
  Learning Materials
  Structured Output from LLMs: Grammars, Regex, and State Machines
  https://www.youtube.com/watch?v=xpvFinvqRCA&t=389s 


  A Guide to Structured Generation Using Constrained Decoding
  https://www.aidancooper.co.uk/constrained-decoding/

  Part 6: Implementing Constrained Decoding
  https://medium.com/@albersj66/part-6-implementing-constrained-decoding-for-phi-3-vision-2c72a1be6a17

  Coalescence: Making LLM Inference 5x Faster
  https://blog.dottxt.ai/coalescence.html

  AI Assistance
  This project made use of AI tools for:

  Structuring README.md

  Brainstorming constrained decoding strategies

