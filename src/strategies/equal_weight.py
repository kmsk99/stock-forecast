"""
동일가중 포트폴리오 전략

모든 종목에 동일한 비중을 할당하는 가장 단순한 전략입니다.
"""

from typing import Optional, Dict, Any
import warnings

import pandas as pd
import numpy as np
from loguru import logger


def weights(
    prices: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """동일가중 포트폴리오 가중치를 계산합니다.
    
    Args:
        prices: 가격 데이터프레임 (인덱스: 날짜, 컬럼: 종목)
        **kwargs: 추가 매개변수 (미사용)
        
    Returns:
        가중치 데이터프레임 (같은 형태, 각 행의 합이 1)
    """
    logger.debug(f"동일가중 전략 실행: {prices.shape[1]} 종목, {prices.shape[0]} 일")
    
    # 종목 수
    n_assets = prices.shape[1]
    
    if n_assets == 0:
        raise ValueError("종목이 없습니다.")
    
    # 동일 가중치 (1/N)
    equal_weight = 1.0 / n_assets
    
    # 모든 날짜와 종목에 동일한 가중치 할당
    weight_matrix = np.full(prices.shape, equal_weight)
    
    # 결측치가 있는 종목은 제외하고 재조정
    result = pd.DataFrame(
        weight_matrix,
        index=prices.index,
        columns=prices.columns
    )
    
    # 각 날짜별로 유효한 종목들에만 가중치 재분배
    for date in result.index:
        valid_mask = prices.loc[date].notna()
        n_valid = valid_mask.sum()
        
        if n_valid > 0:
            # 유효한 종목들에만 동일 가중치
            result.loc[date, :] = 0.0
            result.loc[date, valid_mask] = 1.0 / n_valid
        else:
            # 모든 종목이 결측치인 경우 0으로 설정
            result.loc[date, :] = 0.0
    
    return result


def calculate_portfolio_returns(
    prices: pd.DataFrame,
    weights_df: Optional[pd.DataFrame] = None
) -> pd.Series:
    """포트폴리오 수익률을 계산합니다.
    
    Args:
        prices: 가격 데이터프레임
        weights_df: 가중치 데이터프레임 (None이면 동일가중치 사용)
        
    Returns:
        포트폴리오 수익률 시리즈
    """
    if weights_df is None:
        weights_df = weights(prices)
    
    # 개별 종목 수익률 계산
    returns = prices.pct_change().fillna(0)
    
    # 포트폴리오 수익률 = 가중치 * 개별 수익률의 합
    portfolio_returns = (weights_df * returns).sum(axis=1)
    
    return portfolio_returns


def get_strategy_info() -> Dict[str, Any]:
    """전략 정보를 반환합니다.
    
    Returns:
        전략 정보 딕셔너리
    """
    return {
        'name': 'Equal Weight',
        'description': '모든 종목에 동일한 가중치(1/N)를 할당하는 전략',
        'type': 'static',
        'rebalancing_required': True,
        'parameters': {},
        'advantages': [
            '구현이 간단함',
            '소형주 효과 포착 가능',
            '편향된 집중 투자 방지',
            '낮은 관리 비용'
        ],
        'disadvantages': [
            '시가총액 대비 소형주 과중',
            '거래 비용이 높을 수 있음',
            '개별 종목 리스크 높음'
        ],
        'best_use_cases': [
            '분산투자 목적',
            '소형주 효과 활용',
            '벤치마크 대비 성과',
            '단순한 포트폴리오 구성'
        ]
    }


def validate_inputs(prices: pd.DataFrame) -> bool:
    """입력 데이터를 검증합니다.
    
    Args:
        prices: 가격 데이터프레임
        
    Returns:
        검증 결과 (True: 통과, False: 실패)
    """
    if prices.empty:
        logger.error("가격 데이터가 비어있습니다.")
        return False
    
    if prices.shape[1] == 0:
        logger.error("종목이 없습니다.")
        return False
    
    # 모든 컬럼이 숫자형인지 확인
    non_numeric_cols = []
    for col in prices.columns:
        if not pd.api.types.is_numeric_dtype(prices[col]):
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        logger.warning(f"숫자가 아닌 컬럼이 있습니다: {non_numeric_cols}")
    
    # 결측치 비율 확인
    missing_ratios = prices.isnull().sum() / len(prices)
    high_missing_cols = missing_ratios[missing_ratios > 0.5].index.tolist()
    
    if high_missing_cols:
        logger.warning(f"결측치가 50% 이상인 종목: {high_missing_cols}")
    
    # 가격이 모두 0이거나 음수인 종목 확인
    invalid_price_cols = []
    for col in prices.columns:
        if (prices[col] <= 0).all():
            invalid_price_cols.append(col)
    
    if invalid_price_cols:
        logger.warning(f"유효하지 않은 가격 데이터: {invalid_price_cols}")
    
    return True


def backtest_strategy(
    prices: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    rebalance_freq: str = '1M',
    transaction_cost: float = 0.001
) -> Dict[str, Any]:
    """전략 백테스트를 실행합니다.
    
    Args:
        prices: 가격 데이터프레임
        start_date: 시작 날짜
        end_date: 종료 날짜
        rebalance_freq: 리밸런싱 주기
        transaction_cost: 거래 비용
        
    Returns:
        백테스트 결과 딕셔너리
    """
    logger.info("🧪 동일가중 전략 백테스트 시작")
    
    # 데이터 검증
    if not validate_inputs(prices):
        raise ValueError("입력 데이터 검증 실패")
    
    # 날짜 범위 필터링
    test_data = prices.copy()
    if start_date:
        test_data = test_data[test_data.index >= start_date]
    if end_date:
        test_data = test_data[test_data.index <= end_date]
    
    if test_data.empty:
        raise ValueError("지정된 날짜 범위에 데이터가 없습니다.")
    
    # 가중치 계산
    weights_df = weights(test_data)
    
    # 포트폴리오 수익률 계산
    portfolio_returns = calculate_portfolio_returns(test_data, weights_df)
    
    # 누적 수익률
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # 성과 지표 계산
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    max_dd = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
    
    # 거래 비용 고려 (단순화)
    turnover = weights_df.diff().abs().sum(axis=1).mean()
    net_return = annualized_return - (turnover * transaction_cost * 252)
    
    results = {
        'strategy_name': 'Equal Weight',
        'start_date': test_data.index[0],
        'end_date': test_data.index[-1],
        'total_days': len(test_data),
        'n_assets': test_data.shape[1],
        
        # 수익률 지표
        'total_return': total_return,
        'annualized_return': annualized_return,
        'net_annualized_return': net_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_dd,
        
        # 거래 관련
        'avg_turnover': turnover,
        'estimated_transaction_cost': turnover * transaction_cost * 252,
        
        # 시계열 데이터
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'weights': weights_df,
        
        # 전략 정보
        'strategy_info': get_strategy_info()
    }
    
    logger.success(f"✅ 백테스트 완료 - 연수익률: {annualized_return:.2%}, 샤프비율: {sharpe_ratio:.2f}")
    
    return results


# 테스트 및 예시용 함수들

def create_sample_data(
    n_assets: int = 5,
    n_days: int = 252,
    start_date: str = '2020-01-01'
) -> pd.DataFrame:
    """샘플 가격 데이터를 생성합니다.
    
    Args:
        n_assets: 종목 수
        n_days: 일수
        start_date: 시작 날짜
        
    Returns:
        샘플 가격 데이터프레임
    """
    np.random.seed(42)
    
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    tickers = [f'STOCK_{i:02d}' for i in range(n_assets)]
    
    # 기하 브라운 운동으로 가격 생성
    initial_prices = np.random.uniform(50, 150, n_assets)
    returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))  # 일 수익률
    
    prices = np.zeros((n_days, n_assets))
    prices[0] = initial_prices
    
    for i in range(1, n_days):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    return pd.DataFrame(prices, index=dates, columns=tickers)


def compare_with_benchmark(
    prices: pd.DataFrame,
    benchmark_col: Optional[str] = None
) -> Dict[str, Any]:
    """벤치마크와 성과를 비교합니다.
    
    Args:
        prices: 가격 데이터프레임
        benchmark_col: 벤치마크 종목 컬럼명 (None이면 시가총액 가중 근사)
        
    Returns:
        비교 결과 딕셔너리
    """
    # 동일가중 포트폴리오
    eq_weights = weights(prices)
    eq_returns = calculate_portfolio_returns(prices, eq_weights)
    eq_cumret = (1 + eq_returns).cumprod()
    
    # 벤치마크 (시가총액 가중 근사 또는 지정된 종목)
    if benchmark_col and benchmark_col in prices.columns:
        benchmark_returns = prices[benchmark_col].pct_change().fillna(0)
    else:
        # 시가총액 가중 근사 (첫 번째 종목에 높은 가중치)
        cap_weights = np.array([0.3, 0.2, 0.15, 0.1, 0.05] + [0.2/max(1, prices.shape[1]-5)] * max(0, prices.shape[1]-5))
        cap_weights = cap_weights[:prices.shape[1]]
        cap_weights = cap_weights / cap_weights.sum()
        
        benchmark_returns = (prices.pct_change().fillna(0) * cap_weights).sum(axis=1)
    
    benchmark_cumret = (1 + benchmark_returns).cumprod()
    
    # 성과 비교
    eq_total_return = eq_cumret.iloc[-1] - 1
    bm_total_return = benchmark_cumret.iloc[-1] - 1
    
    excess_return = eq_total_return - bm_total_return
    
    eq_sharpe = (eq_returns.mean() * 252) / (eq_returns.std() * np.sqrt(252))
    bm_sharpe = (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252))
    
    return {
        'equal_weight_return': eq_total_return,
        'benchmark_return': bm_total_return,
        'excess_return': excess_return,
        'equal_weight_sharpe': eq_sharpe,
        'benchmark_sharpe': bm_sharpe,
        'equal_weight_cumulative': eq_cumret,
        'benchmark_cumulative': benchmark_cumret
    }


# CLI 직접 실행용
if __name__ == "__main__":
    # 샘플 데이터로 테스트
    logger.info("🧪 동일가중 전략 테스트 시작")
    
    # 샘플 데이터 생성
    sample_prices = create_sample_data(n_assets=5, n_days=252)
    logger.info(f"📊 샘플 데이터 생성: {sample_prices.shape}")
    
    # 가중치 계산
    sample_weights = weights(sample_prices)
    logger.info(f"⚖️ 가중치 계산 완료: {sample_weights.iloc[0].to_dict()}")
    
    # 백테스트 실행
    backtest_results = backtest_strategy(sample_prices)
    
    print("\n📈 백테스트 결과:")
    print(f"연간 수익률: {backtest_results['annualized_return']:.2%}")
    print(f"변동성: {backtest_results['volatility']:.2%}")
    print(f"샤프 비율: {backtest_results['sharpe_ratio']:.2f}")
    print(f"최대 낙폭: {backtest_results['max_drawdown']:.2%}")
    
    # 벤치마크 비교
    comparison = compare_with_benchmark(sample_prices)
    print(f"\n📊 벤치마크 대비:")
    print(f"초과 수익률: {comparison['excess_return']:.2%}")
    print(f"동일가중 샤프: {comparison['equal_weight_sharpe']:.2f}")
    print(f"벤치마크 샤프: {comparison['benchmark_sharpe']:.2f}")
    
    logger.success("✅ 테스트 완료")