.PHONY: install test lint format typecheck eval data demo notebook run gitleaks clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

install:  ## create the venv and install the package with dev extras
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:  ## unit + integration tests with the 70% coverage gate
	$(PY) -m pytest --cov=src --cov-report=term-missing --cov-fail-under=70

lint:  ## ruff + black --check + mypy --strict
	$(PY) -m ruff check .
	$(PY) -m black --check .
	$(PY) -m mypy --strict src/

format:  ## apply black and ruff fixes
	$(PY) -m black .
	$(PY) -m ruff check --fix .

typecheck:
	$(PY) -m mypy --strict src/

eval:  ## full evaluation -> eval/runs/<date>-*.json, eval/RESULTS.md, README metrics block
	$(PY) -m eval.run --update-readme

data:  ## regenerate the synthetic evaluation set and the data manifest
	$(PY) scripts/generate_synthetic.py
	$(PY) scripts/build_manifest.py

demo:  ## rebuild demo/predictions.json from the heuristic pipeline (no API key needed)
	$(PY) scripts/build_demo_gallery.py

notebook:  ## execute notebooks/demo.ipynb in place (needs the `notebook` extra)
	$(PY) scripts/run_notebook.py

run:  ## start the API on :8000
	$(PY) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

gitleaks:  ## scan the repository for secrets (binary from GitHub Releases)
	gitleaks detect --no-banner --redact --source .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
