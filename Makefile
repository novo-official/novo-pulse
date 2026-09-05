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

.DEFAULT_GOAL := help
.PHONY: help venv install install-optional migrate seed seed-data-only train demo dev \
        backend frontend frontend-install test test-backend test-frontend lint report \
        clean download-models docker-up docker-down check profile validate audit e2e

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

seed: migrate ## Generate synthetic data, train a demo model, publish artefacts
	$(MANAGE) seed_demo --profile $(PROFILE) --horizon $(HORIZON) --publish

seed-data-only: ## Regenerate the synthetic dataset without training
	$(MANAGE) seed_demo --skip-training

train: migrate ## Train on the active data contract
	$(MANAGE) train_forecast --profile $(PROFILE) --horizon $(HORIZON) --metric $(METRIC)

report: ## Show the generated model comparison report
	@cat reports/model_report.md

download-models: ## Pre-fetch optional model weights for offline use
	$(PY) scripts/download_models.py

# ------------------------------------------------------------------ run
demo: install frontend-install seed ## One command: setup -> seed -> ready to run
	@echo ""
	@echo "  Demo data is ready. Start the services with:"
	@echo "      make dev"
	@echo "  then open http://localhost:3000"

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

test-frontend: ## Frontend typecheck, lint and production build
	cd frontend && npx tsc --noEmit && npm run lint && npm run build

audit: ## Live API audit against a running backend (status codes + semantics)
	$(PY) scripts/api_audit.py
	$(PY) scripts/content_audit.py

e2e: ## Browser end-to-end walkthrough (needs both servers running)
	cd frontend && node e2e/ui_drive.mjs
	cd frontend && node e2e/ui_upload.mjs
	cd frontend && node e2e/ui_competition.mjs

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
