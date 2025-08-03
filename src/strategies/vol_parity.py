"""
변동성 패리티 (Volatility Parity) 포트폴리오 전략

각 종목의 변동성에 반비례하여 가중치를 할당하는 전략입니다.
변동성이 낮은 종목에 더 높은 가중치를 부여합니다.
"""

from typing import Optional, Dict, Any, Tuple
import warnings

import pandas as pd
import numpy as np
from loguru import logger


def calculate_volatility(
    prices: pd.DataFrame,
    window: int = 20,
    method: str = 'realized'
) -> pd.DataFrame:
    """변동성을 계산합니다.
    
    Args:
        prices: 가격 데이터프레임
        window: 변동성 계산 윈도우
        method: 변동성 계산 방법 ('realized', 'ewm', 'garch')
        
    Returns:
        변동성 데이터프레임
    """
    returns = prices.pct_change().fillna(0)
    
    if method == 'realized':
        # 실현 변동성 (Rolling Standard Deviation)
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        
    elif method == 'ewm':
        # 지수가중 이동평균 변동성
        volatility = returns.ewm(span=window).std() * np.sqrt(252)
        
    elif method == 'garch':
        # GARCH 모델 (단순화된 버전)
        logger.warning("GARCH 모델은 현재 단순화된 버전으로 구현됩니다.")
        # 간단한 EWMA 변동성으로 대체
        lambda_param = 0.94
        volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for col in returns.columns:
            var_series = pd.Series(index=returns.index, dtype=float)
            var_series.iloc[0] = returns[col].iloc[:window].var()
            
            for i in range(1, len(returns)):
                var_series.iloc[i] = (lambda_param * var_series.iloc[i-1] + 
                                    (1 - lambda_param) * returns[col].iloc[i]**2)
            
            volatility[col] = np.sqrt(var_series * 252)
    
    else:
        raise ValueError(f"지원하지 않는 변동성 계산 방법: {method}")
    
    return volatility


def weights(
    prices: pd.DataFrame,
    vol_window: int = 20,
    vol_method: str = 'realized',
    min_weight: float = 0.01,
    max_weight: float = 0.40,
    target_vol: Optional[float] = None,
    **kwargs
) -> pd.DataFrame:
    """변동성 패리티 포트폴리오 가중치를 계산합니다.
    
    Args:
        prices: 가격 데이터프레임 (인덱스: 날짜, 컬럼: 종목)
        vol_window: 변동성 계산 윈도우
        vol_method: 변동성 계산 방법
        min_weight: 최소 가중치
        max_weight: 최대 가중치
        target_vol: 목표 변동성 (None이면 자동 계산)
        **kwargs: 추가 매개변수
        
    Returns:
        가중치 데이터프레임
    """
    logger.debug(f"변동성 패리티 전략 실행: {prices.shape[1]} 종목, {prices.shape[0]} 일")
    
    # 변동성 계산
    volatilities = calculate_volatility(prices, vol_window, vol_method)
    
    # 가중치 초기화
    weights_df = pd.DataFrame(
        np.zeros(prices.shape),
        index=prices.index,
        columns=prices.columns
    )
    
    for date in weights_df.index:
        # 해당 날짜의 변동성
        date_vols = volatilities.loc[date]
        
        # 유효한 변동성 (NaN이 아니고 양수)
        valid_mask = (date_vols.notna()) & (date_vols > 0)
        
        if valid_mask.sum() == 0:
            continue
            
        valid_vols = date_vols[valid_mask]
        
        # 변동성의 역수로 가중치 계산 (낮은 변동성 = 높은 가중치)
        inv_vols = 1.0 / valid_vols
        
        # 정규화하여 합이 1이 되도록
        raw_weights = inv_vols / inv_vols.sum()
        
        # 가중치 제약 적용
        constrained_weights = apply_weight_constraints(
            raw_weights, min_weight, max_weight
        )
        
        # 결과 저장
        weights_df.loc[date, valid_mask] = constrained_weights
    
    # 목표 변동성 조정
    if target_vol is not None:
        weights_df = adjust_for_target_volatility(
            weights_df, prices, target_vol, vol_window
        )
    
    return weights_df


def apply_weight_constraints(
    raw_weights: pd.Series,
    min_weight: float,
    max_weight: float,
    max_iterations: int = 100
) -> pd.Series:
    """가중치 제약을 적용합니다.
    
    Args:
        raw_weights: 원본 가중치
        min_weight: 최소 가중치
        max_weight: 최대 가중치
        max_iterations: 최대 반복 횟수
        
    Returns:
        제약이 적용된 가중치
    """
    weights = raw_weights.copy()
    
    for _ in range(max_iterations):
        # 최소 가중치 제약
        below_min = weights < min_weight
        if below_min.any():
            excess = (min_weight - weights[below_min]).sum()
            weights[below_min] = min_weight
            
            # 나머지에서 차감
            remaining = weights[~below_min]
            if remaining.sum() > excess:
                weights[~below_min] *= (remaining.sum() - excess) / remaining.sum()
        
        # 최대 가중치 제약
        above_max = weights > max_weight
        if above_max.any():
            excess = (weights[above_max] - max_weight).sum()
            weights[above_max] = max_weight
            
            # 나머지에 분배
            remaining = weights[~above_max]
            if len(remaining) > 0:
                weights[~above_max] += excess / len(remaining)
        
        # 정규화
        weights = weights / weights.sum()
        
        # 수렴 체크
        if not (below_min.any() or above_max.any()):
            break
    
    return weights


def adjust_for_target_volatility(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float,
    vol_window: int
) -> pd.DataFrame:
    """목표 변동성에 맞춰 가중치를 조정합니다.
    
    Args:
        weights_df: 원본 가중치
        prices: 가격 데이터
        target_vol: 목표 변동성 (연율화)
        vol_window: 변동성 계산 윈도우
        
    Returns:
        조정된 가중치
    """
    logger.debug(f"목표 변동성 조정: {target_vol:.2%}")
    
    returns = prices.pct_change().fillna(0)
    adjusted_weights = weights_df.copy()
    
    for i in range(vol_window, len(weights_df)):
        date = weights_df.index[i]
        current_weights = weights_df.loc[date]
        
        if current_weights.sum() == 0:
            continue
            
        # 최근 수익률로 공분산 행렬 계산
        recent_returns = returns.iloc[i-vol_window:i][current_weights > 0]
        
        if len(recent_returns) < vol_window // 2:
            continue
            
        cov_matrix = recent_returns.cov() * 252  # 연율화
        
        # 포트폴리오 변동성 계산
        portfolio_weights = current_weights[current_weights > 0]
        portfolio_vol = np.sqrt(
            portfolio_weights.T @ cov_matrix @ portfolio_weights
        )
        
        if portfolio_vol > 0 and not np.isnan(portfolio_vol):
            # 목표 변동성 비율로 조정
            vol_ratio = target_vol / portfolio_vol
            vol_ratio = np.clip(vol_ratio, 0.5, 2.0)  # 극단적 조정 방지
            
            adjusted_weights.loc[date] = current_weights * vol_ratio
    
    return adjusted_weights


def calculate_risk_metrics(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    window: int = 20
) -> Dict[str, pd.Series]:
    """리스크 지표를 계산합니다.
    
    Args:
        weights_df: 가중치 데이터프레임
        prices: 가격 데이터프레임
        window: 계산 윈도우
        
    Returns:
        리스크 지표 딕셔너리
    """
    returns = prices.pct_change().fillna(0)
    portfolio_returns = (weights_df * returns).sum(axis=1)
    
    metrics = {}
    
    # 포트폴리오 변동성
    metrics['portfolio_volatility'] = portfolio_returns.rolling(
        window=window
    ).std() * np.sqrt(252)
    
    # 개별 종목 기여도 변동성
    vol_contributions = pd.DataFrame(
        index=prices.index, 
        columns=prices.columns
    )
    
    for i in range(window, len(returns)):
        date = returns.index[i]
        w = weights_df.loc[date]
        recent_returns = returns.iloc[i-window:i]
        cov_matrix = recent_returns.cov() * 252
        
        for asset in prices.columns:
            if w[asset] > 0:
                # 자산의 변동성 기여도
                marginal_contrib = (cov_matrix @ w)[asset]
                vol_contributions.loc[date, asset] = w[asset] * marginal_contrib
    
    metrics['volatility_contributions'] = vol_contributions
    
    # 집중도 지표 (Herfindahl Index)
    metrics['concentration'] = (weights_df ** 2).sum(axis=1)
    
    # 유효 종목 수
    metrics['effective_assets'] = 1 / metrics['concentration']
    
    return metrics


def get_strategy_info() -> Dict[str, Any]:
    """전략 정보를 반환합니다.
    
    Returns:
        전략 정보 딕셔너리
    """
    return {
        'name': 'Volatility Parity',
        'description': '각 종목의 변동성에 반비례하여 가중치를 할당하는 리스크 패리티 전략',
        'type': 'dynamic',
        'rebalancing_required': True,
        'parameters': {
            'vol_window': '변동성 계산 윈도우 (기본: 20일)',
            'vol_method': '변동성 계산 방법 (realized, ewm, garch)',
            'min_weight': '최소 가중치 제약 (기본: 1%)',
            'max_weight': '최대 가중치 제약 (기본: 40%)',
            'target_vol': '목표 변동성 (선택사항)'
        },
        'advantages': [
            '변동성 기반 리스크 관리',
            '동적 가중치 조정',
            '하방 리스크 제한',
            '시장 상황 적응성'
        ],
        'disadvantages': [
            '높은 거래 비용 가능성',
            '변동성 예측의 한계',
            '복잡한 구현',
            '과거 데이터 의존성'
        ],
        'best_use_cases': [
            '리스크 관리 중시',
            '변동성이 큰 시장',
            '다양한 자산군 포트폴리오',
            '기관투자자 운용'
        ]
    }


def backtest_strategy(
    prices: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    vol_window: int = 20,
    vol_method: str = 'realized',
    rebalance_freq: str = '1M',
    transaction_cost: float = 0.001,
    **kwargs
) -> Dict[str, Any]:
    """변동성 패리티 전략 백테스트를 실행합니다.
    
    Args:
        prices: 가격 데이터프레임
        start_date: 시작 날짜
        end_date: 종료 날짜
        vol_window: 변동성 계산 윈도우
        vol_method: 변동성 계산 방법
        rebalance_freq: 리밸런싱 주기
        transaction_cost: 거래 비용
        **kwargs: 추가 매개변수
        
    Returns:
        백테스트 결과 딕셔너리
    """
    logger.info("🧪 변동성 패리티 전략 백테스트 시작")
    
    # 날짜 범위 필터링
    test_data = prices.copy()
    if start_date:
        test_data = test_data[test_data.index >= start_date]
    if end_date:
        test_data = test_data[test_data.index <= end_date]
    
    if test_data.empty:
        raise ValueError("지정된 날짜 범위에 데이터가 없습니다.")
    
    # 가중치 계산
    weights_df = weights(
        test_data, 
        vol_window=vol_window, 
        vol_method=vol_method,
        **kwargs
    )
    
    # 포트폴리오 수익률 계산
    returns = test_data.pct_change().fillna(0)
    portfolio_returns = (weights_df * returns).sum(axis=1)
    
    # 누적 수익률
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # 성과 지표 계산
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    max_dd = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
    
    # 거래 비용 계산
    turnover = weights_df.diff().abs().sum(axis=1).mean()
    net_return = annualized_return - (turnover * transaction_cost * 252)
    
    # 리스크 지표
    risk_metrics = calculate_risk_metrics(weights_df, test_data, vol_window)
    
    # 변동성 분석
    vol_analysis = analyze_volatility_targeting(weights_df, test_data, vol_window)
    
    results = {
        'strategy_name': 'Volatility Parity',
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
        
        # 리스크 지표
        'avg_concentration': risk_metrics['concentration'].mean(),
        'avg_effective_assets': risk_metrics['effective_assets'].mean(),
        'volatility_stability': vol_analysis['stability'],
        
        # 시계열 데이터
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'weights': weights_df,
        'risk_metrics': risk_metrics,
        'volatility_analysis': vol_analysis,
        
        # 전략 정보
        'strategy_info': get_strategy_info(),
        'parameters': {
            'vol_window': vol_window,
            'vol_method': vol_method,
            'rebalance_freq': rebalance_freq,
            **kwargs
        }
    }
    
    logger.success(f"✅ 백테스트 완료 - 연수익률: {annualized_return:.2%}, 샤프비율: {sharpe_ratio:.2f}")
    
    return results


def analyze_volatility_targeting(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    window: int
) -> Dict[str, Any]:
    """변동성 타겟팅 효과를 분석합니다.
    
    Args:
        weights_df: 가중치 데이터프레임
        prices: 가격 데이터프레임
        window: 분석 윈도우
        
    Returns:
        분석 결과 딕셔너리
    """
    returns = prices.pct_change().fillna(0)
    portfolio_returns = (weights_df * returns).sum(axis=1)
    
    # 실현 변동성
    realized_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
    
    # 변동성 안정성 (변동성의 변동성)
    vol_of_vol = realized_vol.rolling(window=window).std()
    stability = 1 / (1 + vol_of_vol.mean())  # 높을수록 안정
    
    # 변동성 분포
    vol_stats = {
        'mean': realized_vol.mean(),
        'std': realized_vol.std(),
        'min': realized_vol.min(),
        'max': realized_vol.max(),
        'percentiles': realized_vol.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    }
    
    return {
        'realized_volatility': realized_vol,
        'stability': stability,
        'vol_stats': vol_stats,
        'vol_of_vol': vol_of_vol
    }


# CLI 직접 실행용
if __name__ == "__main__":
    from .equal_weight import create_sample_data
    
    logger.info("🧪 변동성 패리티 전략 테스트 시작")
    
    # 샘플 데이터 생성 (변동성이 다른 종목들)
    np.random.seed(42)
    sample_prices = create_sample_data(n_assets=5, n_days=252)
    
    # 인위적으로 변동성 차이 생성
    vol_multipliers = [0.5, 0.8, 1.0, 1.5, 2.0]  # 종목별 변동성 배수
    
    for i, col in enumerate(sample_prices.columns):
        returns = sample_prices[col].pct_change().fillna(0)
        adjusted_returns = returns * vol_multipliers[i]
        sample_prices[col] = (1 + adjusted_returns).cumprod() * sample_prices[col].iloc[0]
    
    logger.info(f"📊 샘플 데이터 생성: {sample_prices.shape}")
    
    # 가중치 계산
    sample_weights = weights(sample_prices, vol_window=20)
    logger.info(f"⚖️ 가중치 계산 완료")
    
    # 최종 가중치 출력
    final_weights = sample_weights.iloc[-1]
    final_weights = final_weights[final_weights > 0]
    logger.info(f"최종 가중치: {final_weights.to_dict()}")
    
    # 백테스트 실행
    backtest_results = backtest_strategy(sample_prices)
    
    print("\n📈 백테스트 결과:")
    print(f"연간 수익률: {backtest_results['annualized_return']:.2%}")
    print(f"변동성: {backtest_results['volatility']:.2%}")
    print(f"샤프 비율: {backtest_results['sharpe_ratio']:.2f}")
    print(f"최대 낙폭: {backtest_results['max_drawdown']:.2%}")
    print(f"평균 집중도: {backtest_results['avg_concentration']:.3f}")
    print(f"유효 종목 수: {backtest_results['avg_effective_assets']:.1f}")
    
    logger.success("✅ 테스트 완료")