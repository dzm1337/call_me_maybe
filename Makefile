USER ?= $(shell whoami)
GOINFRE := /goinfre/$(USER)

export UV_CACHE_DIR := $(GOINFRE)/.uv_cache
export HF_HOME := $(GOINFRE)/.cache/huggingface
export HF_HUB_CACHE := $(HF_HOME)/hub
export TRANSFORMERS_CACHE := $(HF_HOME)/hub
export HF_DATASETS_CACHE := $(HF_HOME)/datasets

.PHONY: install init_cache run debug clean lint lint-strict

init_cache:
	@mkdir -p "$(UV_CACHE_DIR)"
	@mkdir -p "$(HF_HUB_CACHE)"
	@mkdir -p "$(HF_DATASETS_CACHE)"

install: init_cache
	uv sync

run: init_cache
	uv run python -m src

debug: init_cache
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .uv
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	uv run flake8 .
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
