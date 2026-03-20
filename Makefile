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

.PHONY: help venv setup-core setup-web setup-dev setup-gpu init run web-migrate web-api web-client clean

help:
	@printf '%s\n' \
	  'Perceptrome bootstrap targets:' \
	  '  make setup-core   - install lean CLI/core dependencies into $(VENV)' \
	  '  make setup-web    - install web server deps and client node modules' \
	  '  make setup-dev    - install combined dev/test deps and client node modules' \
	  '  make setup-gpu    - install optional CUDA 12 / GPU training deps' \
	  '  make init         - initialize default local state directories' \
	  '  make run          - run the default CLI command' \
	  '  make web-migrate  - apply Alembic migrations for the web server' \
	  '  make web-api      - start the FastAPI dev server on :8000' \
	  '  make web-client   - start the Vite dev server on :5173' \
	  '  make clean        - remove $(VENV)'

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

clean:
	rm -rf $(VENV)
