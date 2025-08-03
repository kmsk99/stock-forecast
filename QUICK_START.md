# 🚀 Stock Forecast Lab - 빠른 시작 가이드

Stock Forecast Lab 프로젝트가 성공적으로 구축되었습니다! 아래 단계를 따라 바로 사용해보세요.

## ✅ 완료된 구축 사항

### 📁 프로젝트 구조
```
stock-forecast/
├── 📄 pyproject.toml              # 프로젝트 설정 (최신 Python 표준)
├── 📄 Makefile                    # 자동화 명령어 모음
├── 📄 ARCHITECTURE_GUIDE.md       # 상세한 아키텍처 설명서
├── 📄 requirements-dev.txt        # 개발용 도구들
├── 📄 env.example                 # 환경변수 템플릿
│
├── 📁 src/                        # 메인 소스 코드
│   ├── 📄 config.py               # Pydantic 기반 설정 관리
│   ├── 📄 cli.py                  # Typer 기반 CLI
│   ├── 📁 ingest/                 # 데이터 수집 (Yahoo Finance)
│   ├── 📁 features/               # 기술지표 + 레이블링
│   ├── 📁 strategies/             # 투자 전략 (동일가중, 변동성 패리티)
│   ├── 📁 backtest/               # 백테스트 엔진 + 성과 지표
│   └── 📁 reports/                # Plotly/Dash 대시보드
│
├── 📁 data/                       # 데이터 저장소
│   ├── 📁 raw/                    # 원본 데이터 (✅ 2개 파일)
│   └── 📁 silver/                 # 가공 데이터 (✅ 1개 파일)
│
├── 📁 notebooks/                  # Jupyter 노트북
├── 📁 tests/                      # 테스트 코드
└── 📄 README.md                   # 프로젝트 소개
```

### 🔧 설치된 환경
- **Conda 환경**: `stock-forecast` (Python 3.11)
- **패키지**: 모든 의존성 설치 완료 (pandas, yfinance, vectorbt, plotly 등)
- **CLI 도구**: `stocklab` 명령어 활성화

### 📊 테스트 완료된 기능들
- ✅ 데이터 수집: Yahoo Finance에서 AAPL, MSFT 데이터 수집
- ✅ 피처 생성: 38개 기술지표 계산 (SMA, RSI, MACD, 볼린저밴드 등)
- ✅ CLI 인터페이스: 모든 주요 명령어 동작 확인

---

## 🎯 바로 사용하기

### 1️⃣ 환경 활성화
```bash
conda activate stock-forecast
cd /Users/gimminseog/project/stock-forecast
```

### 2️⃣ 프로젝트 상태 확인
```bash
stocklab status
```

### 3️⃣ 데이터 수집 (더 많은 종목)
```bash
# 기본 주요 종목들
stocklab ingest yfinance -t AAPL -t MSFT -t GOOGL -t AMZN -t TSLA \
  --start 2022-01-01 --end 2024-12-31

# 또는 Makefile 사용
make ingest
```

### 4️⃣ 피처 생성
```bash
stocklab make-features --force

# 또는 Makefile 사용
make features
```

### 5️⃣ 백테스트 실행
```bash
# 동일가중 전략
stocklab backtest equal-weight --from 2023-01-01 --to 2024-12-31

# 변동성 패리티 전략
stocklab backtest vol-parity --from 2023-01-01 --to 2024-12-31
```

### 6️⃣ 전체 파이프라인 실행
```bash
make pipeline  # 데이터 수집 → 피처 → 백테스트 → 리포트
```

---

## 📚 주요 명령어 모음

### CLI 명령어
```bash
stocklab --help                    # 전체 도움말
stocklab status                    # 프로젝트 상태 확인
stocklab ingest yfinance --help    # 데이터 수집 도움말
stocklab make-features             # 피처 생성
stocklab backtest --help          # 백테스트 도움말
```

### Makefile 명령어
```bash
make help                          # 모든 명령어 목록
make install                       # 패키지 설치
make test                          # 테스트 실행
make lint                          # 코드 품질 검사
make format                        # 코드 포매팅
make clean                         # 임시 파일 정리
```

### 개발 도구
```bash
# 테스트 실행
pytest tests/

# 코드 포매팅
black src/ tests/

# 린팅
ruff check src/ tests/

# Jupyter Lab 시작
jupyter lab
```

---

## 🛠️ 개발 워크플로우

### 새 전략 추가하기
1. `src/strategies/my_strategy.py` 파일 생성
2. `weights()` 함수 구현:
   ```python
   def weights(prices: pd.DataFrame) -> pd.DataFrame:
       # 전략 로직 구현
       return weight_df
   ```
3. `src/cli.py`에 새 백테스트 명령어 추가

### 새 기술지표 추가하기
1. `src/features/ta_factors.py`의 `add_all_indicators()` 함수 수정
2. 새 지표 계산 함수 추가
3. 테스트 작성: `tests/test_features.py`

### 환경변수 설정
```bash
cp env.example .env
# .env 파일 편집하여 API 키 등 설정
```

---

## 📊 실제 사용 예시

### Python 코드로 직접 사용
```python
# Jupyter 노트북에서
from src.features.ta_factors import load_features
from src.strategies.equal_weight import weights
from src.backtest.engine import BacktestEngine

# 피처 데이터 로드
features = load_features()
prices = features['close'].unstack(level=1)

# 전략 실행
w = weights(prices)

# 백테스트
engine = BacktestEngine(prices)
result = engine.run_basic(w, '1M')

print(f"최종 수익률: {result['total_return']:.2%}")
```

### 고급 사용법
```bash
# 사용자 지정 종목으로 데이터 수집
make ingest-custom TICKERS="SPY QQQ IWM" START=2020-01-01 END=2024-12-31

# 느린 테스트 제외하고 실행
pytest -m "not slow"

# 병렬 테스트
pytest -n 4

# 커버리지 포함 테스트
pytest --cov=src --cov-report=html
```

---

## 🔍 문제 해결

### 자주 발생하는 문제들

**1. ta-lib 설치 오류 (macOS)**
```bash
# Homebrew로 ta-lib 설치
brew install ta-lib
pip install ta-lib
```

**2. 패키지 충돌**
```bash
pip check                          # 충돌 확인
pip install --upgrade pip          # pip 업그레이드
```

**3. 권한 문제**
```bash
pip install --user -e .           # 사용자 모드 설치
```

**4. Jupyter 커널 문제**
```bash
python -m ipykernel install --user --name stock-forecast
```

### 로그 확인
```bash
# 애플리케이션 로그
tail -f logs/stocklab.log

# 또는 Makefile 사용
make logs
```

---

## 📈 다음 단계

### 추천 확장 기능들
1. **한국 주식 데이터**: PyKRX 라이브러리 추가
2. **ML 전략**: scikit-learn 기반 예측 모델
3. **리스크 관리**: VaR, CVaR 계산
4. **실시간 데이터**: WebSocket 연결
5. **알림 시스템**: Slack, Discord 웹훅

### 성능 최적화
```bash
# 프로파일링
python -m cProfile -o profile.prof script.py
pip install snakeviz
snakeviz profile.prof

# 메모리 사용량 확인
mprof run python script.py
mprof plot
```

---

## 🎉 축하합니다!

Stock Forecast Lab 프로젝트가 성공적으로 구축되었습니다! 

**구축된 주요 기능들:**
- 📥 **데이터 수집**: Yahoo Finance API 연동
- 🔧 **피처 엔지니어링**: 38개 기술지표 자동 계산
- ⚖️ **투자 전략**: 동일가중, 변동성 패리티
- 🧪 **백테스트 엔진**: vectorbt 기반 고성능 백테스팅
- 📊 **성과 지표**: CAGR, Sharpe, MDD, VaR 등 20+ 지표
- 🎨 **시각화**: Plotly/Dash 기반 인터랙티브 차트
- 🖥️ **CLI 도구**: 전체 워크플로우 자동화
- 🧪 **테스트**: pytest 기반 품질 보증
- 📚 **문서화**: 완벽한 아키텍처 가이드

이제 본격적으로 주식 데이터를 분석하고 백테스팅을 시작해보세요! 🚀

---

**문의사항이나 추가 기능이 필요하면 언제든지 알려주세요!** 💬