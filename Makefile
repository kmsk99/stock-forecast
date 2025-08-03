.PHONY: help install install-dev test lint format clean ingest features backtest report docker

# 기본 변수
PYTHON := python
PIP := pip
PROJECT_NAME := stock-forecast
SRC_DIR := src
TEST_DIR := tests
DATA_DIR := data

# 기본 타겟 설정
.DEFAULT_GOAL := help

## 도움말
help: ## 사용 가능한 명령어 목록을 표시합니다
	@echo "Stock Forecast Lab - 사용 가능한 명령어:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

## 설치 관련
install: ## 프로젝트 의존성을 설치합니다
	$(PIP) install -e .

install-dev: ## 개발용 의존성을 포함하여 설치합니다
	$(PIP) install -e .[dev,notebook,ml]

install-notebook: ## Jupyter 노트북 지원을 설치합니다
	$(PIP) install -e .[notebook]

## 개발 도구
test: ## 전체 테스트를 실행합니다
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

test-fast: ## 빠른 테스트만 실행합니다 (slow 마커 제외)
	pytest $(TEST_DIR) -v -m "not slow"

lint: ## 코드 린팅을 실행합니다
	ruff check $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR)

format: ## 코드 포매팅을 실행합니다
	black $(SRC_DIR) $(TEST_DIR)
	ruff check $(SRC_DIR) $(TEST_DIR) --fix

format-check: ## 코드 포매팅 검사만 실행합니다
	black --check $(SRC_DIR) $(TEST_DIR)

## 데이터 파이프라인
ingest: ## 기본 주식 데이터를 수집합니다 (AAPL, MSFT, GOOGL, TSLA)
	$(PYTHON) -m $(SRC_DIR).cli ingest yfinance \
		--tickers AAPL MSFT GOOGL TSLA AMZN NVDA META NFLX \
		--start 2020-01-01 \
		--end $$(date +%Y-%m-%d)

ingest-custom: ## 사용자 지정 주식을 수집합니다 (TICKERS, START, END 변수 사용)
	@if [ -z "$(TICKERS)" ]; then \
		echo "사용법: make ingest-custom TICKERS='AAPL MSFT' START=2020-01-01 END=2024-12-31"; \
		exit 1; \
	fi
	$(PYTHON) -m $(SRC_DIR).cli ingest yfinance \
		--tickers $(TICKERS) \
		--start $(START) \
		--end $(END)

features: ## 피처 엔지니어링을 실행합니다
	$(PYTHON) -m $(SRC_DIR).cli make-features \
		--input $(DATA_DIR)/raw \
		--output $(DATA_DIR)/silver

backtest: ## 기본 백테스트를 실행합니다 (동일가중 전략)
	$(PYTHON) -m $(SRC_DIR).cli backtest equal_weight \
		--from 2021-01-01 \
		--to $$(date +%Y-%m-%d)

backtest-all: ## 모든 전략에 대해 백테스트를 실행합니다
	$(PYTHON) -m $(SRC_DIR).cli backtest equal_weight --from 2021-01-01 --to $$(date +%Y-%m-%d)
	$(PYTHON) -m $(SRC_DIR).cli backtest vol_parity --from 2021-01-01 --to $$(date +%Y-%m-%d)

report: ## 최신 백테스트 결과 리포트를 생성합니다
	$(PYTHON) -m $(SRC_DIR).cli report --latest

dashboard: ## 대시보드를 시작합니다
	$(PYTHON) -m $(SRC_DIR).reports.plotly_dash

## 전체 파이프라인
pipeline: ingest features backtest report ## 전체 파이프라인을 실행합니다 (데이터 수집 → 피처 → 백테스트 → 리포트)

## 개발 환경 설정
setup-dev: install-dev ## 개발 환경을 설정합니다
	pre-commit install
	@echo "개발 환경 설정이 완료되었습니다!"

setup-env: ## 환경 파일을 생성합니다
	@if [ ! -f .env ]; then \
		cp env.example .env; \
		echo ".env 파일이 생성되었습니다. 필요한 설정을 편집하세요."; \
	else \
		echo ".env 파일이 이미 존재합니다."; \
	fi

## 정리
clean: ## 임시 파일과 캐시를 정리합니다
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

clean-data: ## 데이터 디렉토리를 정리합니다 (주의: 모든 데이터가 삭제됩니다)
	@echo "경고: 모든 수집된 데이터가 삭제됩니다. 계속하려면 'yes'를 입력하세요:"
	@read confirmation && [ "$$confirmation" = "yes" ] || (echo "취소되었습니다." && exit 1)
	rm -rf $(DATA_DIR)/raw/*
	rm -rf $(DATA_DIR)/silver/*
	@echo "데이터가 정리되었습니다."

## Docker 관련
docker-build: ## Docker 이미지를 빌드합니다
	docker build -t $(PROJECT_NAME) .

docker-run: ## Docker 컨테이너를 실행합니다
	docker run -it --rm -v $$(pwd):/workspace $(PROJECT_NAME)

## Jupyter 노트북
notebook: ## Jupyter Lab을 시작합니다
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

## 로그 확인
logs: ## 최근 로그를 확인합니다
	@if [ -f logs/stocklab.log ]; then \
		tail -f logs/stocklab.log; \
	else \
		echo "로그 파일이 없습니다."; \
	fi

## 상태 확인
status: ## 프로젝트 상태를 확인합니다
	@echo "=== Stock Forecast Lab 상태 ==="
	@echo "Python: $$(python --version)"
	@echo "패키지 설치 상태: $$(pip show stock-forecast >/dev/null 2>&1 && echo '✅ 설치됨' || echo '❌ 미설치')"
	@echo "데이터 디렉토리:"
	@ls -la $(DATA_DIR)/ 2>/dev/null || echo "  데이터 디렉토리가 없습니다."
	@echo "원본 데이터:"
	@ls -la $(DATA_DIR)/raw/ 2>/dev/null || echo "  원본 데이터가 없습니다."
	@echo "가공 데이터:"
	@ls -la $(DATA_DIR)/silver/ 2>/dev/null || echo "  가공 데이터가 없습니다."