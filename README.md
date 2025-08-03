# Stock Forecast Lab 📈

**"데이터 수집 → 피처 가공 → 전략 정의 → 백테스트 → 리포트"**의 전 과정을 순수 파이썬으로 실행할 수 있는 주식 예측 백테스팅 시스템입니다.

## ✨ 주요 특징

- 🔄 **End-to-End 파이프라인**: 데이터 수집부터 성과 분석까지 자동화
- 🧪 **모듈식 설계**: 각 컴포넌트를 독립적으로 테스트하고 교체 가능
- 📊 **다양한 데이터 소스**: yfinance, PyKRX, OpenDART API 지원
- 🚀 **백테스트 엔진**: vectorbt 기반 고성능 백테스팅
- 📈 **대시보드**: Plotly/Dash 기반 인터랙티브 리포트
- 🛠️ **CLI 도구**: 전체 워크플로우를 명령어로 실행

## 🏗️ 프로젝트 구조

```
stock-forecast/
├─ data/                     # 데이터 저장소
│   ├─ raw/                  # 원본 데이터 (yyyy-mm-dd/ticker.csv)
│   └─ silver/               # 가공된 피처 데이터 (parquet)
│
├─ src/                      # 소스 코드
│   ├─ config.py             # 설정 관리 (Pydantic)
│   ├─ cli.py                # CLI 진입점 (Typer)
│   │
│   ├─ utils/                # 공통 유틸리티
│   │   └─ paths.py          # 경로 관리
│   │
│   ├─ ingest/               # 데이터 수집
│   │   ├─ yfinance_cli.py   # Yahoo Finance 수집기
│   │   └─ open_dart.py      # DART 공시 수집기
│   │
│   ├─ features/             # 피처 엔지니어링
│   │   ├─ ta_factors.py     # 기술지표 (SMA, RSI, ATR...)
│   │   └─ labeler.py        # 레이블 생성 (수익률 → 분류/회귀)
│   │
│   ├─ strategies/           # 투자 전략
│   │   ├─ equal_weight.py   # 동일 가중 전략
│   │   ├─ vol_parity.py     # 변동성 패리티
│   │   └─ ml_forecast.py    # ML 예측 기반 전략
│   │
│   ├─ backtest/             # 백테스트 엔진
│   │   ├─ engine.py         # 백테스트 실행기
│   │   └─ metrics.py        # 성과 지표 (CAGR, Sharpe, MDD)
│   │
│   └─ reports/              # 리포트 생성
│       └─ plotly_dash.py    # 대시보드
│
├─ notebooks/                # Jupyter 노트북
├─ tests/                    # 테스트 코드
└─ pyproject.toml            # 프로젝트 설정
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Conda 환경 생성 (이미 완료)
conda activate stock-forecast

# 의존성 설치
pip install -e .

# 개발 도구 설치 (선택)
pip install -e .[dev,notebook]
```

### 2. 설정 파일 생성

```bash
# 환경변수 설정
cp .env.example .env
# .env 파일을 편집하여 API 키 등을 설정
```

### 3. 데이터 수집

```bash
# Yahoo Finance에서 데이터 수집
stocklab ingest yfinance --tickers AAPL MSFT GOOGL --start 2020-01-01 --end 2024-12-31

# 또는 Makefile 사용
make ingest
```

### 4. 피처 생성

```bash
# 기술지표 및 레이블 생성
stocklab make-features --input data/raw --output data/silver

# 또는
make features
```

### 5. 백테스트 실행

```bash
# 동일가중 전략 백테스트
stocklab backtest equal_weight --from 2021-01-01 --to 2024-12-31

# 변동성 패리티 전략
stocklab backtest vol_parity --from 2021-01-01 --to 2024-12-31

# 또는
make backtest
```

### 6. 리포트 생성

```bash
# 백테스트 결과 대시보드
stocklab report --bt-id 20241231T1230

# 또는
make report
```

## 📊 사용 예시

### Python 코드로 직접 실행

```python
from src.ingest.yfinance_cli import collect_data
from src.features.ta_factors import add_technical_indicators
from src.strategies.equal_weight import weights
from src.backtest.engine import run

# 1. 데이터 수집
prices = collect_data(['AAPL', 'MSFT'], '2020-01-01', '2024-12-31')

# 2. 피처 생성
features = add_technical_indicators(prices)

# 3. 전략 실행
w = weights(prices)

# 4. 백테스트
portfolio, metrics = run(prices, weights)
print(f"CAGR: {metrics['cagr']:.2%}")
print(f"Sharpe: {metrics['sharpe']:.2f}")
```

## 🛠️ 개발 가이드

### 새로운 전략 추가

`src/strategies/` 디렉토리에 새 파일을 생성하고 `weights` 함수를 구현:

```python
# src/strategies/my_strategy.py
import pandas as pd

def weights(prices: pd.DataFrame) -> pd.DataFrame:
    """나만의 투자 전략"""
    # 전략 로직 구현
    return weight_df
```

### 테스트 실행

```bash
# 전체 테스트
pytest

# 커버리지 포함
pytest --cov=src

# 특정 모듈만
pytest tests/test_features.py
```

### 코드 품질 검사

```bash
# 포매팅
black src/

# 린팅
ruff check src/

# 타입 체크
mypy src/
```

## 📋 TODO

- [ ] Korean 주식 데이터 (PyKRX) 수집기 구현
- [ ] DART 공시 데이터 파싱 모듈
- [ ] ML 기반 예측 전략 (TimesNet, PatchTST)
- [ ] 리스크 패리티 포트폴리오
- [ ] Streamlit 대시보드 추가
- [ ] Docker 컨테이너화
- [ ] GitHub Actions CI/CD

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이센스

MIT License로 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트 링크: [https://github.com/your-org/stock-forecast](https://github.com/your-org/stock-forecast)