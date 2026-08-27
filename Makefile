.PHONY: help env fetch-small fetch-large fetch-mind data test eval bench submit-mind clean-pyc
.DEFAULT_GOAL := help

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

help:  ## show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $@

env: $(VENV)/bin/activate  ## create .venv and install pinned deps

fetch-small:  ## EB-NeRD demo+small (~0.10 GB) — needed by P1
	./scripts/fetch_data.sh small

fetch-large:  ## EB-NeRD large/test/embeddings (~5.1 GB) — needed by P5
	./scripts/fetch_data.sh large

fetch-mind: env  ## MIND via HuggingFace — requires `hf auth login` first
	./scripts/fetch_data.sh mind

data: env  ## rebuild raw -> unified schema -> temporal split -> feature store (Q1)
	PYTHONPATH=. $(PY) scripts/build_pipeline.py

test: env  ## run the test suite, incl. the no-leakage assertion (Q9)
	PYTHONPATH=. $(VENV)/bin/pytest tests/ -v

submit-mind: env  ## fit popularity, measure on dev, write + validate the MIND submission
	PYTHONPATH=. $(PY) scripts/make_submission_mind.py

eval:  ## metrics table for a predictions file + split (Q4)
	@echo "make eval: not implemented (P2)"; exit 1

bench:  ## index build time, peak RSS, per-query latency (Q6 "breaks at 10x")
	@echo "make bench: not implemented (P5)"; exit 1

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
