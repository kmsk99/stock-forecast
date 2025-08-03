"""
백테스트 엔진 및 성과 지표 테스트
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtest.engine import BacktestEngine, run_backtest
from src.backtest.metrics import (
    calculate_returns_metrics, calculate_risk_adjusted_metrics,
    calculate_drawdown_metrics, calculate_performance_metrics
)
from src.strategies.equal_weight import weights as eq_weights
from src.strategies.vol_parity import weights as vol_weights


@pytest.fixture
def sample_prices():
    """테스트용 샘플 가격 데이터"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    tickers = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']
    
    # 각 종목별로 다른 특성을 가진 가격 생성
    price_data = {}
    base_prices = [100, 150, 200, 80, 120]
    volatilities = [0.15, 0.20, 0.10, 0.25, 0.18]
    
    for i, ticker in enumerate(tickers):
        returns = np.random.normal(0.0005, volatilities[i] / np.sqrt(252), len(dates))
        prices = [base_prices[i]]
        
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        price_data[ticker] = prices
    
    df = pd.DataFrame(price_data, index=dates)
    return df


@pytest.fixture
def sample_weights(sample_prices):
    """테스트용 가중치 데이터"""
    return eq_weights(sample_prices)


@pytest.fixture
def sample_returns():
    """테스트용 수익률 데이터"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = np.random.normal(0.0008, 0.015, len(dates))  # 연 20% 수익률, 24% 변동성
    return pd.Series(returns, index=dates)


class TestBacktestEngine:
    """백테스트 엔진 테스트"""
    
    def test_engine_initialization(self, sample_prices):
        """엔진 초기화 테스트"""
        engine = BacktestEngine(
            prices=sample_prices,
            initial_capital=100000,
            transaction_cost=0.001,
            slippage=0.0002
        )
        
        assert engine.initial_capital == 100000
        assert engine.transaction_cost == 0.001
        assert engine.slippage == 0.0002
        assert len(engine.prices) == len(sample_prices)
        assert len(engine.returns) == len(sample_prices)
    
    def test_basic_backtest(self, sample_prices, sample_weights):
        """기본 백테스트 실행 테스트"""
        engine = BacktestEngine(sample_prices, initial_capital=100000)
        
        # 기본 엔진으로 백테스트 실행
        result = engine.run_basic(sample_weights, rebalance_freq='1M')
        
        # 기본 검증
        assert result['success'] is True
        assert 'portfolio_value' in result
        assert 'portfolio_returns' in result
        assert len(result['portfolio_value']) == len(sample_prices)
        
        # 포트폴리오 가치가 양수인지 확인
        assert (result['portfolio_value'] > 0).all()
        
        # 최종 가치 확인
        assert result['final_value'] > 0
    
    def test_rebalance_dates(self, sample_prices):
        """리밸런싱 날짜 생성 테스트"""
        engine = BacktestEngine(sample_prices)
        
        # 월별 리밸런싱
        monthly_dates = engine._get_rebalance_dates('1M')
        assert len(monthly_dates) > 0
        assert len(monthly_dates) < len(sample_prices)  # 전체 날짜보다 적어야 함
        
        # 일별 리밸런싱
        daily_dates = engine._get_rebalance_dates('1D')
        assert len(daily_dates) == len(sample_prices)
    
    def test_transaction_costs(self, sample_prices, sample_weights):
        """거래 비용 적용 테스트"""
        # 거래 비용 없음
        engine_no_cost = BacktestEngine(sample_prices, transaction_cost=0.0, slippage=0.0)
        result_no_cost = engine_no_cost.run_basic(sample_weights, '1M')
        
        # 거래 비용 있음
        engine_with_cost = BacktestEngine(sample_prices, transaction_cost=0.01, slippage=0.005)
        result_with_cost = engine_with_cost.run_basic(sample_weights, '1M')
        
        # 거래 비용이 있는 경우가 수익률이 낮아야 함
        if result_no_cost['success'] and result_with_cost['success']:
            assert result_with_cost['final_value'] <= result_no_cost['final_value']


class TestPerformanceMetrics:
    """성과 지표 테스트"""
    
    def test_returns_metrics(self, sample_returns):
        """수익률 지표 계산 테스트"""
        metrics = calculate_returns_metrics(sample_returns)
        
        # 필수 지표 확인
        required_metrics = ['total_return', 'cagr', 'annualized_return', 'volatility']
        for metric in required_metrics:
            assert metric in metrics
            assert not pd.isna(metrics[metric])
        
        # 값의 합리성 확인
        assert metrics['total_return'] > -1  # -100% 이상
        assert metrics['volatility'] > 0  # 변동성은 양수
        assert metrics['trading_days'] > 0
    
    def test_risk_adjusted_metrics(self, sample_returns):
        """위험조정 지표 계산 테스트"""
        metrics = calculate_risk_adjusted_metrics(sample_returns, risk_free_rate=0.02)
        
        # 필수 지표 확인
        required_metrics = ['sharpe_ratio', 'sortino_ratio', 'information_ratio']
        for metric in required_metrics:
            assert metric in metrics
            assert not pd.isna(metrics[metric])
        
        # 샤프 비율 범위 확인 (일반적으로 -3 ~ 3 사이)
        assert -5 < metrics['sharpe_ratio'] < 5
    
    def test_drawdown_metrics(self, sample_returns):
        """드로우다운 지표 계산 테스트"""
        metrics = calculate_drawdown_metrics(sample_returns)
        
        # 필수 지표 확인
        required_metrics = ['max_drawdown', 'calmar_ratio', 'drawdown_series']
        for metric in required_metrics:
            assert metric in metrics
        
        # 최대 낙폭은 0 이하여야 함
        assert metrics['max_drawdown'] <= 0
        
        # 드로우다운 시리즈 길이 확인
        assert len(metrics['drawdown_series']) == len(sample_returns)
        
        # 드로우다운은 0 이하여야 함
        assert (metrics['drawdown_series'] <= 0).all()
    
    def test_performance_metrics_integration(self, sample_returns):
        """종합 성과 지표 테스트"""
        metrics = calculate_performance_metrics(sample_returns)
        
        # 주요 지표들이 모두 포함되어 있는지 확인
        expected_categories = [
            'total_return', 'cagr', 'volatility',  # 수익률 지표
            'sharpe_ratio', 'sortino_ratio',  # 위험조정 지표
            'max_drawdown', 'calmar_ratio',  # 드로우다운 지표
            'win_rate', 'profit_factor'  # 일관성 지표
        ]
        
        for metric in expected_categories:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # 메트릭 수가 충분한지 확인 (최소 20개 이상)
        assert len(metrics) >= 20


class TestStrategies:
    """전략 테스트"""
    
    def test_equal_weight_strategy(self, sample_prices):
        """동일가중 전략 테스트"""
        weights = eq_weights(sample_prices)
        
        # 기본 검증
        assert weights.shape == sample_prices.shape
        assert len(weights.columns) == len(sample_prices.columns)
        
        # 가중치 합이 1인지 확인 (유효한 데이터에 대해)
        for date in weights.index[-10:]:  # 마지막 10일만 확인
            row_sum = weights.loc[date].sum()
            if row_sum > 0:  # 유효한 데이터가 있는 경우
                assert abs(row_sum - 1.0) < 1e-10, f"Weight sum: {row_sum}"
        
        # 모든 유효한 가중치가 양수인지 확인
        valid_weights = weights[weights > 0]
        assert (valid_weights >= 0).all().all()
    
    def test_vol_parity_strategy(self, sample_prices):
        """변동성 패리티 전략 테스트"""
        weights = vol_weights(sample_prices, vol_window=20)
        
        # 기본 검증
        assert weights.shape == sample_prices.shape
        
        # 가중치 합 확인 (충분한 데이터가 있는 후반부)
        for date in weights.index[-10:]:
            row_sum = weights.loc[date].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 1e-6, f"Weight sum: {row_sum}"
        
        # 가중치 범위 확인 (0~1 사이)
        assert (weights >= 0).all().all()
        assert (weights <= 1).all().all()


class TestIntegration:
    """통합 테스트"""
    
    @pytest.mark.slow
    def test_full_backtest_pipeline(self):
        """전체 백테스트 파이프라인 테스트 (느린 테스트)"""
        # 샘플 데이터 생성
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        tickers = ['A', 'B', 'C']
        
        prices_data = {}
        for ticker in tickers:
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = [100]
            for r in returns[1:]:
                prices.append(prices[-1] * (1 + r))
            prices_data[ticker] = prices
        
        prices_df = pd.DataFrame(prices_data, index=dates)
        
        # 백테스트 엔진 테스트
        engine = BacktestEngine(prices_df, initial_capital=10000)
        weights = eq_weights(prices_df)
        
        result = engine.run_basic(weights, '1W')  # 주간 리밸런싱
        
        # 결과 검증
        assert result['success'] is True
        assert result['final_value'] > 0
        
        # 성과 지표 계산
        if 'portfolio_returns' in result:
            metrics = calculate_performance_metrics(result['portfolio_returns'])
            assert len(metrics) > 10  # 충분한 지표가 계산되어야 함
    
    def test_strategy_comparison(self, sample_prices):
        """전략 비교 테스트"""
        # 동일가중 전략
        eq_weights_df = eq_weights(sample_prices)
        engine_eq = BacktestEngine(sample_prices)
        result_eq = engine_eq.run_basic(eq_weights_df, '1M')
        
        # 변동성 패리티 전략
        vol_weights_df = vol_weights(sample_prices, vol_window=20)
        engine_vol = BacktestEngine(sample_prices)
        result_vol = engine_vol.run_basic(vol_weights_df, '1M')
        
        # 두 전략 모두 성공해야 함
        assert result_eq['success'] is True
        assert result_vol['success'] is True
        
        # 두 전략 모두 양의 최종 가치를 가져야 함
        assert result_eq['final_value'] > 0
        assert result_vol['final_value'] > 0
        
        # 수익률 시리즈 길이가 같아야 함
        assert len(result_eq['portfolio_returns']) == len(result_vol['portfolio_returns'])


# 엣지 케이스 테스트
class TestEdgeCases:
    """엣지 케이스 테스트"""
    
    def test_empty_data(self):
        """빈 데이터 처리 테스트"""
        empty_prices = pd.DataFrame()
        
        with pytest.raises((ValueError, IndexError)):
            BacktestEngine(empty_prices)
    
    def test_single_asset(self):
        """단일 자산 테스트"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        single_asset = pd.DataFrame({
            'SINGLE': np.random.uniform(90, 110, len(dates))
        }, index=dates)
        
        engine = BacktestEngine(single_asset)
        weights = eq_weights(single_asset)
        result = engine.run_basic(weights, '1M')
        
        assert result['success'] is True
        # 단일 자산의 가중치는 1이어야 함
        assert abs(weights.iloc[-1, 0] - 1.0) < 1e-10
    
    def test_missing_prices(self):
        """결측 가격 데이터 처리 테스트"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices_with_nan = pd.DataFrame({
            'A': np.random.uniform(90, 110, len(dates)),
            'B': np.random.uniform(90, 110, len(dates))
        }, index=dates)
        
        # 일부 데이터를 NaN으로 설정
        prices_with_nan.iloc[10:20, 0] = np.nan
        
        engine = BacktestEngine(prices_with_nan)
        weights = eq_weights(prices_with_nan)
        result = engine.run_basic(weights, '1M')
        
        # 결측 데이터가 있어도 백테스트는 성공해야 함
        assert result['success'] is True


# 실행 시 테스트
if __name__ == "__main__":
    print("🧪 백테스트 테스트 시작")
    
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'A': np.random.uniform(90, 110, len(dates)),
        'B': np.random.uniform(80, 120, len(dates)),
        'C': np.random.uniform(95, 105, len(dates))
    }, index=dates)
    
    print("📊 백테스트 엔진 테스트...")
    engine = BacktestEngine(sample_data, initial_capital=10000)
    weights = eq_weights(sample_data)
    result = engine.run_basic(weights, '1M')
    
    print(f"✅ 백테스트 완료: 최종 가치 ${result['final_value']:.2f}")
    
    print("📈 성과 지표 계산...")
    if 'portfolio_returns' in result:
        metrics = calculate_performance_metrics(result['portfolio_returns'])
        print(f"✅ 성과 지표 계산 완료: {len(metrics)} 지표")
        print(f"   총 수익률: {metrics.get('total_return', 0):.2%}")
        print(f"   샤프 비율: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"   최대 낙폭: {metrics.get('max_drawdown', 0):.2%}")
    
    print("✅ 테스트 완료")