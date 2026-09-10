# ---------------------------------------------------------------------------
# Novo Pulse
#
#   make demo     one command from a fresh clone to a working dashboard
#   make dev      run backend + frontend in the foreground
#   make test     backend tests + frontend typecheck/lint/build
# ---------------------------------------------------------------------------

SHELL := /bin/bash
VENV  := .venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip
MANAGE := $(PY) backend/manage.py

PROFILE ?= demo
HORIZON ?= 90
METRIC  ?= wape
JURY_RUN_DIR ?= artifacts/pol4_jury_reproduction
HISTORY_RUN_DIR ?= artifacts/pol4_history66_reproduction

.DEFAULT_GOAL := help
.PHONY: help venv install install-optional migrate train dev \
        backend frontend frontend-install test test-backend test-frontend lint report \
        clean download-models docker-up docker-down check profile validate audit e2e \
        pol4 pol4-baseline pol4-ablation pol4-submission pol4-predict \
        pol4-cluster pol4-cluster-quick \
        pol4-predict-raw pol4-predict-calibrated test-pol4

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup
venv: ## Create the Python virtualenv
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PIP) install -q --upgrade pip

install: venv ## Install backend dependencies
	$(PIP) install -q -r requirements.txt
	@echo "backend dependencies installed"

install-optional: venv ## Install Chronos / NeuralForecast / Optuna (large)
	$(PIP) install -r requirements-optional.txt

frontend-install: ## Install frontend dependencies
	cd frontend && npm install --no-audit --no-fund

# --------------------------------------------------------------- Pol 4
# The competition pipeline. Separate from `train` on purpose: it models the
# two-clock (log_date / checkin) problem and writes the submission file, while
# `train` runs the generic single-axis platform on whatever the contract says.
pol4: ## Pol 4: champion backtest + stability + results.csv (needs data/raw/pol4/)
	PYTHONPATH=backend $(PY) -m ml.pol4.pipeline

pol4-baseline: ## Pol 4: the pickup baseline alone - fast, and the fallback
	PYTHONPATH=backend $(PY) -m ml.pol4.pipeline --model baseline

pol4-ablation: ## Pol 4: rerun the full feature ladder and rewrite experiments.csv
	PYTHONPATH=backend $(PY) -m ml.pol4.pipeline --ablation

pol4-submission: ## Pol 4: skip the backtest, just regenerate results.csv
	PYTHONPATH=backend $(PY) -m ml.pol4.pipeline --skip-backtest --skip-stability

pol4-cluster: ## Pol 4: sweep the aggregation level, then write a clustered submission
	PYTHONPATH=backend $(PY) -m ml.pol4.cluster_pipeline --levels 6

pol4-cluster-quick: ## Pol 4: the same sweep on one window, for a smoke test
	PYTHONPATH=backend $(PY) -m ml.pol4.cluster_pipeline --levels 3 --cutoffs 2025-10-22 --trees 200

pol4-predict: pol4-predict-calibrated ## Pol 4: predict with the selected calibrated variant

pol4-predict-raw: ## Pol 4: predict with the raw, uncalibrated LightGBM variant
	PYTHONPATH=backend $(PY) -m ml.pol4.inference --variant raw

pol4-predict-calibrated: ## Pol 4: predict with guarded calibration
	PYTHONPATH=backend $(PY) -m ml.pol4.inference --variant calibrated

test-pol4: ## Run only the Pol 4 test suite
	$(PY) -m pytest backend/tests -q -k pol4

.PHONY: install-locked pol4-jury pol4-jury-experiments pol4-recover-trainset pol4-uncertainty test-jury-ui pol4-pitch pol4-event-study pol4-history-ablation
install-locked: venv ## Install the validated Linux/Python environment snapshot
	$(PIP) install -r requirements.lock

pol4-jury: ## Generate verified evidence and the destination review queue
	PYTHONPATH=backend $(PY) -m ml.pol4.jury

pol4-jury-experiments: ## Run six audit challengers on the same five city-level folds
	PYTHONPATH=backend $(PY) -m ml.pol4.jury_experiments --output $(JURY_RUN_DIR)

pol4-recover-trainset: ## Rebuild the champion trainset and compare original hashes
	$(PY) scripts/recover_pol4_trainset.py

pol4-uncertainty: ## Evaluate experimental residual bands on completed challenger OOF folds
	PYTHONPATH=backend $(PY) -m ml.pol4.uncertainty

pol4-history-ablation: ## Direct 66-feature history audit on the original five cutoffs
	PYTHONPATH=backend $(PY) -m ml.pol4.jury_experiments --arms champion66 --output $(HISTORY_RUN_DIR)/champion
	PYTHONPATH=backend $(PY) -m ml.pol4.jury_experiments --arms origin66 --output $(HISTORY_RUN_DIR)/origin

pol4-event-study: ## Descriptive shock-window and seasonality diagnostic
	$(PY) scripts/pol4_event_study.py

pol4-pitch: pol4-jury ## Generate the offline seven-minute pitch with Q&A appendix
	$(PY) scripts/build_pol4_pitch.py

test-jury-ui: ## Browser checks for the dashboard and evidence room (servers running)
	cd frontend && node e2e/jury.mjs

# ------------------------------------------------------------- database
migrate: ## Apply database migrations
	$(MANAGE) migrate --noinput

check: ## Django system checks
	$(MANAGE) check

# ----------------------------------------------------------------- data
profile: ## Profile a dataset and suggest a mapping (FILE=path/to/data.csv)
	@test -n "$(FILE)" || (echo "usage: make profile FILE=data/raw/your.csv" && exit 1)
	$(MANAGE) profile_dataset --file $(FILE)

validate: ## Data-quality + leakage report for the active contract
	$(MANAGE) validate_dataset

train: migrate ## Train on the active data contract
	$(MANAGE) train_forecast --profile $(PROFILE) --horizon $(HORIZON) --metric $(METRIC)

report: ## Show the generated model comparison report
	@cat reports/model_report.md

download-models: ## Pre-fetch optional model weights for offline use
	$(PY) scripts/download_models.py

# ------------------------------------------------------------------ run
demo: install frontend-install migrate pol4 ## One command: setup -> Pol 4 forecast -> ready to run
	@echo "Pol 4 pipeline complete. Run 'make dev' to open the dashboard."

dev: ## Run backend and frontend together
	@trap 'kill 0' EXIT; \
	$(MANAGE) runserver 0.0.0.0:8000 & \
	cd frontend && npm run dev & \
	wait

backend: ## Run only the backend
	$(MANAGE) runserver 0.0.0.0:8000

frontend: ## Run only the frontend
	cd frontend && npm run dev

# ---------------------------------------------------------------- tests
test: test-backend test-frontend ## Run everything

test-backend: ## Backend test suite
	$(PY) -m pytest

test-frontend: ## Frontend unit tests, typecheck, lint and production build
	cd frontend && npm test && npx tsc --noEmit && npm run lint && npm run build

audit: ## Live API audit against a running backend (status codes + semantics)
	$(PY) scripts/api_audit.py
	$(PY) scripts/content_audit.py

e2e: ## Browser end-to-end walkthrough (needs both servers running)
	cd frontend && node e2e/ui_drive.mjs

lint: ## Lint the frontend
	cd frontend && npm run lint

# ---------------------------------------------------------------- docker
docker-up: ## Build and start the Docker stack
	docker compose up --build

docker-down: ## Stop the Docker stack
	docker compose down

# --------------------------------------------------------------- cleanup
clean: ## Remove artefacts, caches and build output
	rm -rf runs/* reports/*.md data/uploads/* frontend/.next
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned"
