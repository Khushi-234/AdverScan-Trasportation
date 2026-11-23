.PHONY: help install dev-install format lint test clean run-ui run-api

help:
	@echo "AdverScan Management Commands"
	@echo "-----------------------------"
	@echo "make install      : Install production dependencies"
	@echo "make dev-install  : Install development dependencies in editable mode"
	@echo "make run-ui       : Launch Streamlit Web Application"
	@echo "make run-api      : Start FastAPI REST server"
	@echo "make format       : Format code using Black and isort"
	@echo "make lint         : Run code linting (flake8, mypy)"
	@echo "make test         : Run tests using pytest"
	@echo "make clean        : Remove temporary files and caches"

install:
	pip install -r requirements.txt

dev-install:
	pip install -e .[dev]

run-ui:
	streamlit run frontend/streamlit_app.py

run-api:
	uvicorn app.api.main:app --reload

format:
	black app/ frontend/ tests/
	isort app/ frontend/ tests/

lint:
	flake8 app/ frontend/ tests/
	mypy app/

test:
	pytest tests/ --cov=app

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage coverage.xml htmlcov *.egg-info build dist
