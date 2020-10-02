export SHELL := /bin/bash

test:
	pytest -n auto --doctest-modules --cov=pyleoclim --cov-config=.coveragerc pyleoclim

unittests:
	pytest -n auto --cov=pyleoclim --cov-config=.coveragerc pyleoclim

lint:
	flake8 pyleoclim
