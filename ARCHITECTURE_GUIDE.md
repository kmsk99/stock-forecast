# 🏗️ Stock Forecast Lab 아키텍처 가이드

이 문서는 Stock Forecast Lab 프로젝트의 구조, 설정 파일들, 그리고 개발 도구들에 대해 자세히 설명합니다. Python에는 익숙하지만 프로젝트 관리 도구들이 낯선 개발자들을 위해 작성되었습니다.

## 📋 목차

1. [프로젝트 구조 개요](#프로젝트-구조-개요)
2. [핵심 설정 파일들](#핵심-설정-파일들)
3. [빌드 및 의존성 관리](#빌드-및-의존성-관리)
4. [개발 도구들](#개발-도구들)
5. [CI/CD 및 자동화](#cicd-및-자동화)
6. [환경 설정 및 구성](#환경-설정-및-구성)
7. [테스트 및 품질 관리](#테스트-및-품질-관리)

---

## 프로젝트 구조 개요

### 🗂️ 디렉토리 구조

```
stock-forecast/
├── 📁 data/                    # 데이터 저장소
│   ├── 📁 raw/                 # 원본 데이터 (CSV, JSON 등)
│   └── 📁 silver/              # 가공된 데이터 (Parquet 등)
│
├── 📁 src/                     # 메인 소스 코드
│   ├── 📄 config.py            # 설정 관리 (Pydantic)
│   ├── 📄 cli.py               # CLI 진입점 (Typer)
│   │
│   ├── 📁 utils/               # 공통 유틸리티
│   ├── 📁 ingest/              # 데이터 수집 모듈
│   ├── 📁 features/            # 피처 엔지니어링
│   ├── 📁 strategies/          # 투자 전략
│   ├── 📁 backtest/            # 백테스트 엔진
│   └── 📁 reports/             # 리포트 생성
│
├── 📁 notebooks/               # Jupyter 노트북
├── 📁 tests/                   # 테스트 코드
│
├── 📄 pyproject.toml           # 프로젝트 설정 (최신 표준)
├── 📄 requirements-dev.txt     # 개발용 패키지
├── 📄 Makefile                 # 자동화 명령어
├── 📄 env.example              # 환경변수 템플릿
└── 📄 README.md                # 프로젝트 소개
```

### 🧩 모듈 구조 설명

**데이터 계층 (Data Layer)**
- `raw/`: Yahoo Finance, DART 등에서 수집한 원본 데이터
- `silver/`: 기술지표가 추가된 분석용 데이터

**비즈니스 로직 계층 (Business Layer)**  
- `ingest/`: 외부 API에서 데이터 수집
- `features/`: 기술지표 계산 및 레이블 생성
- `strategies/`: 투자 전략 구현
- `backtest/`: 성과 측정 및 백테스트

**프레젠테이션 계층 (Presentation Layer)**
- `reports/`: 차트, 대시보드, HTML 리포트
- `cli.py`: 명령줄 인터페이스

---

## 핵심 설정 파일들

### 📄 pyproject.toml - 모던 Python 프로젝트 설정

`pyproject.toml`은 Python 프로젝트의 **최신 표준 설정 파일**입니다. 기존의 `setup.py`, `requirements.txt`, `setup.cfg` 등을 하나로 통합합니다.

#### 🔍 주요 섹션 분석

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```
- **목적**: 패키지를 빌드할 때 사용할 도구 지정
- **설명**: `setuptools`와 `wheel`을 사용해 배포 가능한 패키지 생성

```toml
[project]
name = "stock-forecast"
version = "0.1.0"
description = "주식 예측 백테스트 시스템"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    # ... 더 많은 패키지들
]
```
- **목적**: 프로젝트 메타데이터 및 의존성 정의
- **dependencies**: 프로젝트 실행에 필요한 필수 패키지들
- **버전 지정**: `>=2.0.0`은 "2.0.0 이상" 의미

```toml
[project.optional-dependencies]
dev = ["pytest>=7.4.0", "black>=23.0.0"]
ml = ["scikit-learn>=1.3.0", "torch>=2.1.0"]
```
- **목적**: 선택적 의존성 그룹
- **dev**: 개발 시에만 필요한 도구들
- **ml**: 머신러닝 기능 사용 시에만 필요

```toml
[project.scripts]
stocklab = "src.cli:app"
```
- **목적**: CLI 명령어 등록
- **효과**: `pip install -e .` 후 `stocklab` 명령어로 실행 가능

```toml
[tool.black]
line-length = 88
target-version = ['py311']
```
- **목적**: 코드 포매터 `black` 설정
- **line-length**: 한 줄 최대 문자 수
- **target-version**: 대상 Python 버전

#### 🚀 사용법

```bash
# 프로젝트 설치 (개발 모드)
pip install -e .

# 개발 도구 포함 설치
pip install -e .[dev]

# 특정 그룹만 설치
pip install -e .[ml,notebook]
```

### 📄 requirements-dev.txt - 개발 도구 의존성

**`pyproject.toml`과의 차이점:**
- `pyproject.toml`: 프로젝트 **실행**에 필요한 패키지
- `requirements-dev.txt`: **개발**에만 필요한 패키지 (더 세분화)

```bash
# 개발 환경 설정
pip install -r requirements-dev.txt
```

**주요 도구들:**
- `pytest`: 테스트 프레임워크
- `black`: 코드 포매터  
- `ruff`: 빠른 린터 (flake8 대체)
- `mypy`: 정적 타입 검사
- `pre-commit`: Git 커밋 전 자동 검사

---

## 빌드 및 의존성 관리

### 🐍 가상환경 관리

**Conda 환경 (현재 사용 중):**
```bash
# 환경 생성
conda create -n stock-forecast python=3.11 -y

# 환경 활성화
conda activate stock-forecast

# 패키지 설치
pip install -e .
```

**대안 - 순수 Python venv:**
```bash
# 가상환경 생성
python -m venv venv

# 활성화 (macOS/Linux)
source venv/bin/activate

# 활성화 (Windows)
venv\Scripts\activate
```

### 📦 패키지 버전 관리

**버전 지정 방식:**
- `pandas>=2.0.0`: 2.0.0 이상 (권장)
- `pandas==2.1.0`: 정확히 2.1.0 (엄격)
- `pandas~=2.1.0`: 2.1.x 버전 (호환성)
- `pandas>=2.0.0,<3.0.0`: 범위 지정

**의존성 충돌 해결:**
```bash
# 의존성 트리 확인
pip install pipdeptree
pipdeptree

# 충돌 검사
pip check
```

---

## 개발 도구들

### 🔨 Makefile - 자동화 명령어

`Makefile`은 **반복적인 작업을 자동화**하는 도구입니다. 원래는 C/C++ 컴파일용이지만, 현재는 다양한 언어에서 작업 자동화에 사용됩니다.

#### 📝 Makefile 문법 기초

```makefile
target: dependencies
	command
	another_command
```

**주의사항:**
- 들여쓰기는 반드시 **탭(Tab)**을 사용 (스페이스 안됨!)
- 각 명령어는 별도 프로세스에서 실행됨

#### 🎯 주요 타겟들

```makefile
help: ## 도움말 표시
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
```
- `help`: 사용 가능한 명령어 목록 표시
- `@`: 명령어 자체는 출력하지 않음
- `##`: 도움말 설명

```makefile
install: ## 프로젝트 의존성을 설치합니다
	$(PIP) install -e .
```
- `$(PIP)`: 변수 사용 (파일 상단에서 `PIP := pip` 정의)

```makefile
test: ## 전체 테스트를 실행합니다
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR)
```

#### 🚀 Makefile 사용법

```bash
# 도움말 보기
make help

# 패키지 설치
make install

# 테스트 실행
make test

# 데이터 수집
make ingest

# 전체 파이프라인 실행
make pipeline
```

#### 💡 Makefile의 장점

1. **표준화**: 팀원 모두가 동일한 명령어 사용
2. **자동화**: 복잡한 명령어 조합을 간단하게
3. **문서화**: 명령어가 곧 문서 역할
4. **의존성**: 타겟 간 의존 관계 관리

### 🎨 코드 품질 도구

#### Black - 코드 포매터
```bash
# 코드 포매팅
black src/ tests/

# 검사만 (변경 안함)
black --check src/
```

**특징:**
- "타협 없는" 포매터 (설정 최소화)
- 일관된 코드 스타일 강제
- Git 커밋 시 자동 실행 가능

#### Ruff - 린터 (Python 3.11+)
```bash
# 린팅 검사
ruff check src/

# 자동 수정 가능한 문제 수정
ruff check src/ --fix
```

**장점:**
- **매우 빠름** (Rust로 작성)
- flake8, isort, pylint 등을 하나로 통합
- 500+ 규칙 지원

#### MyPy - 정적 타입 검사
```bash
# 타입 검사
mypy src/
```

**사용 예시:**
```python
def calculate_return(price: float, prev_price: float) -> float:
    return (price - prev_price) / prev_price

# 잘못된 사용 - mypy가 에러 감지
result = calculate_return("100", "90")  # str는 float가 아님
```

---

## CI/CD 및 자동화

### 🔄 Pre-commit Hooks

Git 커밋하기 전에 자동으로 코드 검사를 실행합니다.

#### 설정 파일 `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.287
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
```

#### 사용법

```bash
# 설치
pip install pre-commit

# 훅 설치
pre-commit install

# 수동 실행
pre-commit run --all-files
```

**동작 방식:**
1. `git commit` 실행
2. Pre-commit이 자동으로 black, ruff 실행
3. 문제가 있으면 커밋 중단
4. 수정 후 다시 커밋

### 🚀 GitHub Actions (예시)

`.github/workflows/test.yml`:

```yaml
name: Test and Lint

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: pytest --cov=src
    
    - name: Run linting
      run: |
        black --check src/
        ruff check src/
```

---

## 환경 설정 및 구성

### 🔧 환경 변수 관리

#### env.example - 환경변수 템플릿

```bash
# API 키 설정
YAHOO_FINANCE_API_KEY=your_key_here
DART_API_KEY=your_dart_key

# 데이터 경로
DATA_DIR=/path/to/data
RAW_DATA_DIR=${DATA_DIR}/raw

# 로깅 설정
LOG_LEVEL=INFO
LOG_FILE=logs/stocklab.log
```

#### 사용법

```bash
# 템플릿 복사
cp env.example .env

# .env 파일 편집 (실제 값 입력)
nano .env
```

**보안 주의사항:**
- `.env` 파일은 **Git에 커밋하지 않음** (`.gitignore`에 포함)
- `env.example`만 커밋하여 템플릿 공유

#### src/config.py - 설정 관리

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str = ""
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**Pydantic Settings 장점:**
- 자동 타입 변환
- 환경변수 자동 로드
- 기본값 설정
- 유효성 검사

---

## 테스트 및 품질 관리

### 🧪 PyTest - 테스트 프레임워크

#### 테스트 구조

```
tests/
├── test_features.py        # 피처 엔지니어링 테스트
├── test_backtest.py        # 백테스트 테스트
├── test_strategies.py      # 전략 테스트
└── conftest.py            # 공통 fixture
```

#### 주요 기능들

**Fixture - 테스트 데이터 준비**
```python
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'close': [100, 101, 99, 102]
    })

def test_calculate_returns(sample_data):
    # 테스트 로직
    pass
```

**매개변수화 테스트**
```python
@pytest.mark.parametrize("window,expected", [
    (5, [None, None, None, None, 100.4]),
    (10, [None] * 9 + [100.6])
])
def test_sma(window, expected):
    # 여러 케이스를 한 번에 테스트
    pass
```

**느린 테스트 마킹**
```python
@pytest.mark.slow
def test_full_backtest():
    # 시간이 오래 걸리는 테스트
    pass
```

#### 실행 방법

```bash
# 전체 테스트
pytest

# 특정 파일
pytest tests/test_features.py

# 느린 테스트 제외
pytest -m "not slow"

# 커버리지 포함
pytest --cov=src --cov-report=html

# 병렬 실행
pytest -n 4
```

### 📊 코드 커버리지

커버리지는 **테스트가 코드의 몇 %를 실행했는지** 측정합니다.

```bash
# 커버리지 실행
pytest --cov=src --cov-report=html

# HTML 리포트 확인
open htmlcov/index.html
```

**목표:**
- **80% 이상**: 좋은 커버리지
- **90% 이상**: 매우 좋은 커버리지
- **100%**: 완벽하지만 과도할 수 있음

---

## 🛠️ 개발 워크플로우

### 일반적인 개발 과정

1. **환경 설정**
   ```bash
   conda activate stock-forecast
   make install-dev
   ```

2. **기능 개발**
   ```bash
   # 새 브랜치 생성
   git checkout -b feature/new-strategy
   
   # 코드 작성
   # ...
   ```

3. **테스트 실행**
   ```bash
   make test
   make lint
   ```

4. **커밋 및 푸시**
   ```bash
   git add .
   git commit -m "Add new volatility strategy"  # pre-commit 자동 실행
   git push origin feature/new-strategy
   ```

5. **Pull Request 생성**

### 🔄 일상적인 작업들

```bash
# 새로운 전략 테스트
make ingest                    # 데이터 수집
make features                  # 피처 생성
make backtest                  # 백테스트 실행
make report                    # 리포트 생성

# 코드 품질 관리
make format                    # 코드 포매팅
make lint                      # 린팅 검사
make test                      # 테스트 실행

# 전체 파이프라인
make pipeline                  # 모든 과정을 순서대로
```

---

## 🚀 고급 주제들

### Docker 컨테이너화

**Dockerfile 예시:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install -e .

COPY src/ src/
CMD ["stocklab", "dashboard"]
```

### 성능 최적화

**프로파일링:**
```bash
# 라인별 성능 측정
kernprof -l -v slow_function.py

# 메모리 사용량 측정
mprof run python script.py
mprof plot
```

### 배포 전략

**패키지 배포:**
```bash
# PyPI 업로드용 빌드
python -m build

# TestPyPI에 업로드
twine upload --repository testpypi dist/*
```

---

## 🎯 핵심 도구 선택 기준

| 도구 | 목적 | 대안 | 선택 이유 |
|------|------|------|-----------|
| **pyproject.toml** | 프로젝트 설정 | setup.py | 최신 표준, 통합 관리 |
| **Makefile** | 작업 자동화 | Scripts | 표준적, 문서화 효과 |
| **Black** | 코드 포매팅 | autopep8 | 타협 없는 스타일 |
| **Ruff** | 린팅 | flake8 | 속도, 통합성 |
| **pytest** | 테스트 | unittest | 풍부한 기능, 생태계 |
| **Pydantic** | 설정 관리 | configparser | 타입 안전성, 검증 |

---

## 📚 추가 학습 자료

### 필수 문서들
- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

### 추천 도서
- "Architecture Patterns with Python" - 대규모 Python 프로젝트 구조
- "Effective Python" - Python 모범 사례

### 유용한 도구들
- **Poetry**: pyproject.toml 기반 의존성 관리
- **Hatch**: 최신 Python 프로젝트 관리 도구
- **Nox**: 다중 환경 테스트 자동화

---

## ❓ 자주 묻는 질문

### Q: pyproject.toml vs requirements.txt?
A: `pyproject.toml`이 **최신 표준**입니다. requirements.txt는 레거시 방식이지만 여전히 널리 사용됩니다.

### Q: Makefile을 왜 사용하나요?
A: 복잡한 명령어를 **표준화**하고 **문서화**하기 위해서입니다. 팀원 모두가 동일한 방식으로 작업할 수 있습니다.

### Q: 테스트는 얼마나 작성해야 하나요?
A: **핵심 로직**은 반드시 테스트하고, 전체 커버리지 80% 이상을 목표로 합니다.

### Q: 코드 스타일은 어떻게 통일하나요?
A: **Black + Ruff + Pre-commit**을 사용하면 자동으로 일관된 스타일이 유지됩니다.

---

## 🎉 마무리

이 가이드를 통해 Stock Forecast Lab 프로젝트의 구조와 도구들을 이해하셨기를 바랍니다. 각 도구는 개발 생산성과 코드 품질을 높이기 위해 신중히 선택되었습니다.

**개발 시 권장사항:**
1. 새 기능 개발 전 `make test` 실행
2. 커밋 전 `make lint` 실행  
3. 복잡한 로직에는 테스트 작성
4. 환경변수는 `.env` 파일로 관리
5. 새로운 의존성은 pyproject.toml에 추가

더 궁금한 점이 있으면 언제든지 문의하세요! 🚀