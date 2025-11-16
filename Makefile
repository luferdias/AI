# Makefile for Crack Detection Pipeline

.PHONY: help install test lint format clean docker-build docker-run setup

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make setup         - Run setup script"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean temporary files"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run with Docker Compose"

install:
	pip install -r requirements.txt

setup:
	python setup.py

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/ --max-line-length=100
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

docker-build:
	cd deployments/docker && docker build -t crack-detection-api -f Dockerfile ../..

docker-run:
	cd deployments/docker && docker-compose up -d

docker-stop:
	cd deployments/docker && docker-compose down

train-unet:
	python scripts/train.py --config configs/unet_config.yaml --model unet

train-yolo:
	python scripts/train.py --config configs/yolo_data.yaml --model yolo

api-dev:
	python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
