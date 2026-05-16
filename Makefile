SHELL := /usr/bin/env bash

VENV ?= .venv
PYTHON ?= python3
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
PERCEPTROME := $(VENV)/bin/perceptrome
UVICORN := $(VENV)/bin/uvicorn
RUN_ARGS ?= catalog-show config/plasmids_100.txt
WEB_CLIENT_DIR := perceptrome_web/client
WEB_SERVER_DIR := perceptrome_web/server
ALEMBIC := $(VENV)/bin/python -m alembic -c $(WEB_SERVER_DIR)/alembic.ini

.DEFAULT_GOAL := help

.PHONY: help venv setup-core setup-web setup-web-interactive setup-dev setup-gpu init run web-migrate web-api web-client web-test web-test-server web-test-client docker-dev docker-prod docker-prod-local docker-build clean

help:
	@printf '%s\n' \
	  'Perceptrome bootstrap targets:' \
	  '  make setup-core            - install lean CLI/core dependencies into $(VENV)' \
	  '  make setup-web             - install web server deps and client node modules' \
	  '  make setup-web-interactive - interactive setup wizard (dev or AWS prod)' \
	  '  make setup-dev             - install combined dev/test deps and client node modules' \
	  '  make setup-gpu             - install optional CUDA 12 / GPU training deps' \
	  '  make init                  - initialize default local state directories' \
	  '  make run                   - run the default CLI command' \
	  '  make web-migrate           - apply Alembic migrations for the web server' \
	  '  make web-api               - start the FastAPI dev server on :8000' \
	  '  make web-client            - start the Vite dev server on :5173' \
	  '  make web-test              - run web server and client tests' \
	  '  make web-test-server       - run FastAPI/Alembic tests with bootstrap disabled' \
	  '  make web-test-client       - run Vite client typecheck/tests' \
	  '  make docker-dev            - start local dev stack via Docker Compose' \
	  '  make docker-prod           - build and start production stack' \
	  '  make docker-prod-local     - prod stack + bundled postgres (local preview)' \
	  '  make docker-build          - build production Docker image only' \
	  '  make clean                 - remove $(VENV)'

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

setup-core: venv
	$(PIP) install -r requirements/core.txt

setup-web: venv
	$(PIP) install -r requirements/web.txt
	npm install --prefix $(WEB_CLIENT_DIR)


setup-dev: venv
	$(PIP) install -r requirements/dev.txt
	npm install --prefix $(WEB_CLIENT_DIR)

setup-gpu: venv
	$(PIP) install -r requirements/gpu-cu12.txt

init: setup-core
	$(PERCEPTROME) --config config/stream_config.yaml init

run: init
	$(PERCEPTROME) $(RUN_ARGS)

web-migrate:
	$(ALEMBIC) upgrade head

web-api:
	cd $(WEB_SERVER_DIR) && $(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

web-client:
	npm run dev --prefix $(WEB_CLIENT_DIR)

web-test: web-test-server web-test-client

web-test-server:
	BOOTSTRAP_ADMIN_EMAIL= BOOTSTRAP_ADMIN_PASSWORD= $(PYTEST) $(WEB_SERVER_DIR)/tests -q

web-test-client:
	npm --prefix $(WEB_CLIENT_DIR) run test

setup-web-interactive:
	$(PYTHON) setup_web.py

docker-dev:
	docker compose -f docker-compose.dev.yml up

docker-prod:
	docker compose -f docker-compose.prod.yml up -d

docker-prod-local:
	docker compose -f docker-compose.prod.yml -f docker-compose.prod.local.yml up -d --build

docker-build:
	docker compose -f docker-compose.prod.yml build

clean:
	rm -rf $(VENV)
