.PHONY: help env fetch-small fetch-testset fetch-large fetch-mind data check-data test eval bench clean-pyc
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

fetch-testset:  ## EB-NeRD test set only (~1.5 GB) — the minimum `make data` needs for EB-NeRD
	./scripts/fetch_data.sh testset

fetch-large:  ## EB-NeRD large/test/embeddings (~5.1 GB) — needed by P5
	./scripts/fetch_data.sh large

fetch-mind: env  ## MIND via HuggingFace — requires `hf auth login` first
	./scripts/fetch_data.sh mind

data: env  ## rebuild raw -> unified schema -> temporal split -> feature store (Q1)
	PYTHONPATH=. $(PY) scripts/build_pipeline.py

check-data: env  ## print the data facts behind the Phase 1 feature definitions (RESULTS.md Q1)
	PYTHONPATH=. $(PY) scripts/check_phase1_data.py

test: env  ## run the test suite, incl. the no-leakage assertion (Q9)
	PYTHONPATH=. $(VENV)/bin/pytest tests/ -v

eval:  ## full two-stage metrics, slices, bootstrap CIs (A2 Q5)
	@echo "make eval: not implemented (A2 P5)"; exit 1

bench:  ## index memory, p99 latency, cost/QPS (A2 Q4)
	@echo "make bench: not implemented (A2 P4)"; exit 1

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
