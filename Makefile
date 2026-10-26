.PHONY: install install-app test lint format geometry faces detection bench studio clean

install:
	pip install -e .

install-app:
	pip install -e ".[app]"

test:
	pytest -q

lint:
	ruff check src apps benchmarks scripts tests

format:
	ruff format src apps benchmarks scripts tests

geometry:
	python benchmarks/run_geometry.py

faces:
	python benchmarks/run_faces.py

detection:
	python -m cvkit detection tune

bench: geometry faces

studio:
	streamlit run apps/studio.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
