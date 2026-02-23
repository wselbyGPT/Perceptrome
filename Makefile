SHELL := /usr/bin/env bash

VENV ?= .venv
PYTHON ?= python3
PIP := $(VENV)/bin/pip
PERCEPTROME := $(VENV)/bin/perceptrome
RUN_ARGS ?= catalog-show config/plasmids_100.txt

.DEFAULT_GOAL := run

.PHONY: setup init run clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

init: setup
	$(PERCEPTROME) init --config config/stream_config.yaml

run: init
	$(PERCEPTROME) $(RUN_ARGS)

clean:
	rm -rf $(VENV)
