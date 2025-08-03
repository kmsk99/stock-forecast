"""
피처 엔지니어링 모듈 테스트
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.ta_factors import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_atr, add_all_indicators,
    process_single_ticker
)
from src.features.labeler import (
    calculate_forward_returns, create_classification_labels,
    create_regression_labels, add_all_labels
)


@pytest.fixture
def sample_price_data():
    """테스트용 샘플 가격 데이터를 생성합니다."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # 기하 브라운 운동으로 가격 생성
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [100]
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    data = pd.DataFrame({
        'date': dates,
        'ticker': 'TEST',
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    return data


class TestTechnicalIndicators:
    """기술지표 테스트 클래스"""
    
    def test_calculate_sma(self, sample_price_data):
        """SMA 계산 테스트"""
        sma = calculate_sma(sample_price_data, window=10)
        
        # 기본 검증
        assert len(sma) == len(sample_price_data)
        assert sma.isna().sum() == 9  # 첫 9개는 NaN
        assert sma.iloc[10:].isna().sum() == 0  # 나머지는 모두 값이 있어야 함
        
        # 값 범위 검증
        close_prices = sample_price_data['close']
        valid_sma = sma.dropna()
        assert valid_sma.min() >= close_prices.min() * 0.8  # 합리적 범위
        assert valid_sma.max() <= close_prices.max() * 1.2
    
    def test_calculate_ema(self, sample_price_data):
        """EMA 계산 테스트"""
        ema = calculate_ema(sample_price_data, window=10)
        
        # 기본 검증
        assert len(ema) == len(sample_price_data)
        # EMA는 첫 번째 값부터 계산 가능
        assert not ema.iloc[1:].isna().any()
    
    def test_calculate_rsi(self, sample_price_data):
        """RSI 계산 테스트"""
        rsi = calculate_rsi(sample_price_data, window=14)
        
        # 기본 검증
        assert len(rsi) == len(sample_price_data)
        
        # RSI 값 범위 검증 (0-100)
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert valid_rsi.min() >= 0
            assert valid_rsi.max() <= 100
    
    def test_calculate_macd(self, sample_price_data):
        """MACD 계산 테스트"""
        macd_dict = calculate_macd(sample_price_data)
        
        # 기본 검증
        required_keys = ['macd', 'macd_signal', 'macd_histogram']
        for key in required_keys:
            assert key in macd_dict
            assert len(macd_dict[key]) == len(sample_price_data)
        
        # MACD 히스토그램 = MACD - Signal
        macd = macd_dict['macd'].dropna()
        signal = macd_dict['macd_signal'].dropna()
        histogram = macd_dict['macd_histogram'].dropna()
        
        if len(macd) > 0 and len(signal) > 0:
            common_idx = macd.index.intersection(signal.index)
            if len(common_idx) > 0:
                np.testing.assert_array_almost_equal(
                    (macd.loc[common_idx] - signal.loc[common_idx]).values,
                    histogram.loc[common_idx].values,
                    decimal=10
                )
    
    def test_calculate_bollinger_bands(self, sample_price_data):
        """볼린저 밴드 계산 테스트"""
        bb_dict = calculate_bollinger_bands(sample_price_data)
        
        # 기본 검증
        required_keys = ['bb_upper', 'bb_middle', 'bb_lower']
        for key in required_keys:
            assert key in bb_dict
            assert len(bb_dict[key]) == len(sample_price_data)
        
        # 밴드 순서 검증 (Upper > Middle > Lower)
        upper = bb_dict['bb_upper'].dropna()
        middle = bb_dict['bb_middle'].dropna()
        lower = bb_dict['bb_lower'].dropna()
        
        if len(upper) > 0 and len(middle) > 0 and len(lower) > 0:
            common_idx = upper.index.intersection(middle.index).intersection(lower.index)
            if len(common_idx) > 10:  # 충분한 데이터가 있을 때만
                assert (upper.loc[common_idx] >= middle.loc[common_idx]).all()
                assert (middle.loc[common_idx] >= lower.loc[common_idx]).all()
    
    def test_add_all_indicators(self, sample_price_data):
        """모든 지표 추가 테스트"""
        enhanced_data = add_all_indicators(sample_price_data)
        
        # 기본 검증
        assert len(enhanced_data) <= len(sample_price_data)  # 결측치 제거로 줄어들 수 있음
        assert len(enhanced_data.columns) > len(sample_price_data.columns)  # 컬럼이 추가되어야 함
        
        # 주요 지표들이 포함되어 있는지 확인
        expected_indicators = ['sma_20', 'ema_20', 'rsi', 'macd', 'bb_upper', 'atr']
        for indicator in expected_indicators:
            assert indicator in enhanced_data.columns


class TestLabeling:
    """레이블링 테스트 클래스"""
    
    @pytest.fixture
    def sample_multiindex_data(self):
        """멀티인덱스 테스트 데이터"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        tickers = ['STOCK_A', 'STOCK_B']
        
        data_list = []
        for ticker in tickers:
            prices = np.random.uniform(50, 150, len(dates))
            for i, date in enumerate(dates):
                data_list.append({
                    'date': date,
                    'ticker': ticker,
                    'close': prices[i]
                })
        
        df = pd.DataFrame(data_list)
        return df.set_index(['date', 'ticker'])
    
    def test_calculate_forward_returns(self, sample_price_data):
        """미래 수익률 계산 테스트"""
        forward_data = calculate_forward_returns(sample_price_data, periods=[1, 5])
        
        # 기본 검증
        assert 'forward_return_1d' in forward_data.columns
        assert 'forward_return_5d' in forward_data.columns
        assert 'forward_log_return_1d' in forward_data.columns
        assert 'forward_log_return_5d' in forward_data.columns
        
        # 마지막 데이터들은 NaN이어야 함
        assert forward_data['forward_return_1d'].iloc[-1:].isna().all()
        assert forward_data['forward_return_5d'].iloc[-5:].isna().all()
    
    def test_create_classification_labels(self, sample_price_data):
        """분류 레이블 생성 테스트"""
        # 먼저 미래 수익률 계산
        forward_data = calculate_forward_returns(sample_price_data, periods=[5])
        
        # 분위수 기반 분류
        quantile_labels = create_classification_labels(
            forward_data, 'forward_return_5d', method='quantile', n_classes=3
        )
        
        # 기본 검증
        assert len(quantile_labels) == len(forward_data)
        
        # 유효한 레이블 확인
        valid_labels = quantile_labels.dropna()
        if len(valid_labels) > 0:
            unique_labels = set(valid_labels.astype(str))
            assert len(unique_labels) <= 3  # 최대 3개 클래스
        
        # 임계값 기반 분류
        threshold_labels = create_classification_labels(
            forward_data, 'forward_return_5d', method='threshold', thresholds=[-0.02, 0.02]
        )
        
        valid_threshold = threshold_labels.dropna()
        if len(valid_threshold) > 0:
            unique_threshold = set(valid_threshold.astype(str))
            assert len(unique_threshold) <= 3  # 3개 구간
    
    def test_create_regression_labels(self, sample_price_data):
        """회귀 레이블 생성 테스트"""
        # 미래 수익률 계산
        forward_data = calculate_forward_returns(sample_price_data, periods=[5])
        
        # 회귀 레이블 생성
        reg_labels = create_regression_labels(
            forward_data, 'forward_return_5d', normalize=True
        )
        
        # 기본 검증
        assert len(reg_labels) == len(forward_data)
        
        # 정규화 검증
        valid_labels = reg_labels.dropna()
        if len(valid_labels) > 10:  # 충분한 데이터가 있을 때
            # 평균이 0에 가까워야 함 (정규화됨)
            assert abs(valid_labels.mean()) < 0.1
            # 표준편차가 1에 가까워야 함
            assert abs(valid_labels.std() - 1) < 0.1
    
    def test_add_all_labels(self, sample_price_data):
        """모든 레이블 추가 테스트"""
        labeled_data = add_all_labels(sample_price_data, periods=[1, 5])
        
        # 기본 검증
        assert len(labeled_data.columns) > len(sample_price_data.columns)
        
        # 미래 수익률 컬럼 확인
        forward_cols = [col for col in labeled_data.columns if col.startswith('forward_return_')]
        assert len(forward_cols) >= 2  # 1d, 5d
        
        # 레이블 컬럼 확인
        label_cols = [col for col in labeled_data.columns if col.startswith('label_')]
        assert len(label_cols) > 0  # 적어도 하나의 레이블이 있어야 함


class TestIntegration:
    """통합 테스트"""
    
    def test_full_pipeline(self, sample_price_data):
        """전체 파이프라인 테스트"""
        # 1. 기술지표 추가
        enhanced_data = add_all_indicators(sample_price_data)
        assert len(enhanced_data) > 0
        
        # 2. 레이블 추가
        final_data = add_all_labels(enhanced_data, periods=[1, 5])
        assert len(final_data) > 0
        
        # 3. 최종 데이터 검증
        assert len(final_data.columns) > len(sample_price_data.columns) * 2
        
        # 4. 결측치 비율 확인 (너무 많으면 안됨)
        missing_ratio = final_data.isnull().sum().sum() / (len(final_data) * len(final_data.columns))
        assert missing_ratio < 0.5  # 50% 미만이어야 함
    
    def test_process_single_ticker(self, sample_price_data):
        """단일 종목 처리 테스트"""
        processed_data = process_single_ticker(sample_price_data)
        
        # 기본 검증
        assert len(processed_data) > 0
        assert len(processed_data.columns) > len(sample_price_data.columns)
        
        # 날짜 정렬 확인
        assert processed_data['date'].is_monotonic_increasing
        
        # 필수 컬럼 확인
        required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in processed_data.columns


# 실행 시 테스트
if __name__ == "__main__":
    # pytest를 사용하지 않을 때의 간단한 테스트
    print("🧪 피처 엔지니어링 테스트 시작")
    
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = np.random.uniform(90, 110, 100)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'ticker': 'TEST',
        'open': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # 기술지표 테스트
    print("📊 기술지표 계산...")
    enhanced = add_all_indicators(sample_data)
    print(f"✅ 기술지표 추가 완료: {len(enhanced.columns)} 컬럼")
    
    # 레이블링 테스트
    print("🏷️ 레이블 생성...")
    labeled = add_all_labels(enhanced, periods=[1, 5])
    print(f"✅ 레이블 생성 완료: {len(labeled.columns)} 컬럼")
    
    print("✅ 테스트 완료")