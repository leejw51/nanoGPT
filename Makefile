# Makefile for nanoGPT — quick commands for download / train / inference.
# Defaults are macOS/CPU friendly (no CUDA). Override any variable on the CLI, e.g.:
#   make train DEVICE=mps
#   make inference INIT_FROM=gpt2 START="Once upon a time"

# ---- load .env if present (see .env.example) ----
# `-include` = don't error when .env is missing. Lines are KEY=value, so they
# override the ?= defaults below; `export` passes them to the python process too.
-include .env
export

# ---- configuration (override on the command line) ----
# NOTE: keep comments on their own lines — a trailing "# ..." would otherwise
# become part of the value (Make includes the spaces before the #).
PYTHON         ?= python
# DEPS: python packages installed by `make install` (matches the README)
DEPS           ?= torch numpy transformers datasets tiktoken wandb tqdm requests
DATASET        ?= shakespeare_char
CONFIG         ?= config/train_shakespeare_char.py
OUT_DIR        ?= out-shakespeare-char
# DEVICE: cpu | cuda | mps
DEVICE         ?= cpu
# COMPILE: True needs CUDA/Triton; keep False on Mac
COMPILE        ?= False
# GPT2 variant: gpt2 | gpt2-medium | gpt2-large | gpt2-xl
GPT2           ?= gpt2
# INIT_FROM: resume (local ckpt) | gpt2* (pretrained)
INIT_FROM      ?= resume
NUM_SAMPLES    ?= 5
MAX_NEW_TOKENS ?= 500
# TEMPERATURE: <1.0 safer/sharper, >1.0 wilder
TEMPERATURE    ?= 0.8
# TOP_K: keep only the top_k most likely tokens when sampling
TOP_K          ?= 200
# START: optional prompt for inference (empty = model default)
START          ?=

TRAIN_BIN := data/$(DATASET)/train.bin

.DEFAULT_GOAL := help
.PHONY: help install download download-model data prepare train inference interactive clean

help: ## Show this help (default)
	@echo ""
	@echo "  nanoGPT Makefile"
	@echo "  ----------------"
	@echo "  make             Show this help"
	@echo "  make install     Install python dependencies with pip"
	@echo "  make download    Download the dataset ($(DATASET)), then print its location + size"
	@echo "  make train       Prepare data + train a model  (config: $(CONFIG))"
	@echo "  make inference   Generate text from a trained or pretrained model"
	@echo "  make interactive REPL: type a prompt, get a response, loop"
	@echo ""
	@echo "  Helpers:"
	@echo "  make download-model  Download pretrained GPT-2 weights ($(GPT2)) into the HuggingFace cache"
	@echo "  make data            Prepare/tokenize the dataset ($(DATASET))"
	@echo "  make clean           Remove the training output dir ($(OUT_DIR))"
	@echo ""
	@echo "  Override variables, e.g. 'make train DEVICE=mps':"
	@echo "    DEVICE=$(DEVICE)        CONFIG=$(CONFIG)"
	@echo "    DATASET=$(DATASET)   OUT_DIR=$(OUT_DIR)   COMPILE=$(COMPILE)"
	@echo "    INIT_FROM=$(INIT_FROM)        GPT2=$(GPT2)"
	@echo "    NUM_SAMPLES=$(NUM_SAMPLES)          MAX_NEW_TOKENS=$(MAX_NEW_TOKENS)          START='$(START)'"
	@echo ""
	@echo "  Typical flows:"
	@echo "    # train your own tiny model, then sample from it"
	@echo "    make train  &&  make inference"
	@echo ""
	@echo "    # or skip training and use OpenAI's pretrained GPT-2"
	@echo "    make download-model GPT2=gpt2-xl"
	@echo "    make inference INIT_FROM=gpt2-xl START=\"The meaning of life is\""
	@echo ""

install: ## Install python dependencies with pip
	@echo ">> Installing dependencies with pip: $(DEPS)"
	$(PYTHON) -m pip install $(DEPS)

download: data ## Download the dataset, then print its location and size
	@echo ""
	@echo ">> Dataset '$(DATASET)' downloaded to:"
	@echo "     $(abspath data/$(DATASET))"
	@echo ">> Files:"
	@ls -lh data/$(DATASET)/*.txt data/$(DATASET)/*.bin data/$(DATASET)/*.pkl 2>/dev/null || true
	@echo ">> Total size:"
	@du -sh data/$(DATASET)
	@echo ""

download-model: ## Download pretrained GPT-2 weights (no training needed)
	@echo ">> Downloading pretrained model '$(GPT2)' (cached under ~/.cache/huggingface)..."
	$(PYTHON) -c "from model import GPT; GPT.from_pretrained('$(GPT2)')"
	@echo ">> Done. Try: make inference INIT_FROM=$(GPT2)"

data: ## Prepare/tokenize the dataset if not already done
	@if [ -f "$(TRAIN_BIN)" ]; then \
		echo ">> Dataset '$(DATASET)' already prepared ($(TRAIN_BIN))."; \
	else \
		echo ">> Preparing dataset '$(DATASET)'..."; \
		$(PYTHON) data/$(DATASET)/prepare.py; \
	fi

prepare: data ## Alias for 'make data'

train: data ## Train a model from the chosen config (streams live progress)
	@echo ">> Training $(CONFIG) on $(DEVICE) (out_dir=$(OUT_DIR))..."
	@echo ">> Live progress prints each log_interval: 'iter X/max (pct%): loss ...'  (Ctrl-C to stop)"
	$(PYTHON) -u train.py $(CONFIG) \
		--device=$(DEVICE) --compile=$(COMPILE) \
		--out_dir=$(OUT_DIR) --dataset=$(DATASET)
	@echo ">> Training complete. Checkpoint (if val improved): $(OUT_DIR)/ckpt.pt"
	@echo ">> Next: make inference"

inference: ## Generate text (resume -> OUT_DIR ckpt; gpt2* -> pretrained)
	@echo ">> Sampling from init_from=$(INIT_FROM) on $(DEVICE)..."
	$(PYTHON) sample.py --init_from=$(INIT_FROM) --out_dir=$(OUT_DIR) \
		--device=$(DEVICE) --num_samples=$(NUM_SAMPLES) \
		--max_new_tokens=$(MAX_NEW_TOKENS) --temperature=$(TEMPERATURE) --top_k=$(TOP_K) \
		$(if $(strip $(START)),--start="$(START)",)

interactive: ## REPL: enter a prompt, print the response, loop (:q to quit)
	@echo ">> Interactive REPL  (init_from=$(INIT_FROM), device=$(DEVICE)) — :q to quit"
	$(PYTHON) -u interactive.py --init_from=$(INIT_FROM) --out_dir=$(OUT_DIR) \
		--device=$(DEVICE) --max_new_tokens=$(MAX_NEW_TOKENS) \
		--temperature=$(TEMPERATURE) --top_k=$(TOP_K)

clean: ## Remove the training output directory
	@echo ">> Removing $(OUT_DIR)..."
	rm -rf $(OUT_DIR)
