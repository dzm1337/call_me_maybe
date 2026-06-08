USER ?= $(shell whoami)

# 1. Procura se existe algum goinfre no sistema
GOINFRE_DETECTED := $(firstword $(wildcard /goinfre/$(USER) /nfs/goinfre/$(USER) /sgoinfre/$(USER)))

# 2. Se o goinfre existir, injeta as variáveis. Em casa (onde não existe), ignora este bloco.
ifneq ($(GOINFRE_DETECTED),)
    init_cache install run debug lint lint-strict: export UV_CACHE_DIR := $(GOINFRE_DETECTED)/.uv_cache
    init_cache install run debug lint lint-strict: export HF_HOME := $(GOINFRE_DETECTED)/.cache/huggingface
    init_cache install run debug lint lint-strict: export HF_HUB_CACHE := $(GOINFRE_DETECTED)/.cache/huggingface/hub
    init_cache install run debug lint lint-strict: export TRANSFORMERS_CACHE := $(GOINFRE_DETECTED)/.cache/huggingface/hub
    init_cache install run debug lint lint-strict: export HF_DATASETS_CACHE := $(GOINFRE_DETECTED)/.cache/huggingface/datasets
endif

.PHONY: install init_cache run debug clean lint lint-strict

init_cache:
ifneq ($(GOINFRE_DETECTED),)
	@echo "[42 Mode] Initializing cache folders in: $(GOINFRE_DETECTED)"
	@mkdir -p "$(UV_CACHE_DIR)"
	@mkdir -p "$(HF_HUB_CACHE)"
	@mkdir -p "$(HF_DATASETS_CACHE)"
	@echo "Cache folders initialized!"
else
	@echo "[Home Mode] No goinfre detected. Using your PC's default local storage. No action needed."
endif

install: 
	uv sync

run:
	uv run python -m src

debug:
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