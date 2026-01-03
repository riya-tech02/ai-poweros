.PHONY: help install install-dev setup test lint format clean run docker-build docker-up docker-down

help:
	@echo "AI-PowerOS Development Commands"
	@echo "================================"
	@echo "install          : Install production dependencies"
	@echo "install-dev      : Install development dependencies"
	@echo "setup            : Complete setup"
	@echo "test             : Run all tests"
	@echo "test-unit        : Run unit tests"
	@echo "lint             : Run linters"
	@echo "format           : Format code"
	@echo "clean            : Clean temporary files"
	@echo "run              : Run API server"
	@echo "docker-build     : Build Docker image"
	@echo "docker-up        : Start all services"
	@echo "docker-down      : Stop all services"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

setup: install-dev
	@echo "Setting up AI-PowerOS..."
	python scripts/init_neo4j.py || true
	python scripts/create_kafka_topics.py || true

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

lint:
	flake8 src/ tests/ || true
	mypy src/ || true

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t ai-poweros/api:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down -v

logs:
	docker-compose logs -f
