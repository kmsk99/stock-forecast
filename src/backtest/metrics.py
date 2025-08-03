"""
백테스트 성과 지표 계산 모듈

CAGR, Sharpe, MDD, CVaR 등 다양한 성과 지표를 계산합니다.
"""

from typing import Dict, Any, Optional, List, Tuple
import warnings

import pandas as pd
import numpy as np
from loguru import logger


def calculate_returns_metrics(returns: pd.Series) -> Dict[str, float]:
    """수익률 기반 지표를 계산합니다.
    
    Args:
        returns: 일일 수익률 시리즈
        
    Returns:
        수익률 지표 딕셔너리
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    # 기본 통계
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    
    # 연율화 (252 거래일 기준)
    trading_days = 252
    years = n_periods / trading_days
    
    # CAGR (Compound Annual Growth Rate)
    if years > 0 and total_return > -1:
        cagr = (1 + total_return) ** (1 / years) - 1
    else:
        cagr = 0.0
    
    # Annualized Return (산술평균 방식)
    annualized_return = returns.mean() * trading_days
    
    # 변동성 (Volatility)
    volatility = returns.std() * np.sqrt(trading_days)
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'trading_days': n_periods,
        'years': years
    }


def calculate_risk_adjusted_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """위험조정 지표를 계산합니다.
    
    Args:
        returns: 일일 수익률 시리즈
        risk_free_rate: 무위험 수익률 (연율)
        
    Returns:
        위험조정 지표 딕셔너리
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    # 기본 지표
    annualized_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    if volatility > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    else:
        sharpe_ratio = 0.0
    
    # Sortino Ratio (하방 변동성만 고려)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_deviation = downside_returns.std() * np.sqrt(252)
        if downside_deviation > 0:
            sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation
        else:
            sortino_ratio = np.inf
    else:
        sortino_ratio = np.inf
    
    # Information Ratio (대 벤치마크, 여기서는 무위험수익률 대비)
    excess_returns = returns - (risk_free_rate / 252)
    tracking_error = excess_returns.std() * np.sqrt(252)
    
    if tracking_error > 0:
        information_ratio = excess_returns.mean() * 252 / tracking_error
    else:
        information_ratio = 0.0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'information_ratio': information_ratio,
        'tracking_error': tracking_error,
        'downside_deviation': downside_deviation if len(downside_returns) > 0 else 0.0
    }


def calculate_drawdown_metrics(returns: pd.Series) -> Dict[str, Any]:
    """드로우다운 지표를 계산합니다.
    
    Args:
        returns: 일일 수익률 시리즈
        
    Returns:
        드로우다운 지표 딕셔너리
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    # 누적 수익률
    cumulative_returns = (1 + returns).cumprod()
    
    # 최고점 (Running Maximum)
    rolling_max = cumulative_returns.expanding().max()
    
    # 드로우다운
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    
    # Maximum Drawdown
    max_drawdown = drawdown.min()
    
    # Maximum Drawdown 기간
    max_dd_start = None
    max_dd_end = None
    max_dd_duration = 0
    
    if max_drawdown < 0:
        max_dd_idx = drawdown.idxmin()
        
        # 시작점 찾기 (최고점)
        start_candidates = rolling_max[:max_dd_idx]
        if len(start_candidates) > 0:
            max_dd_start = start_candidates[start_candidates == start_candidates.max()].index[-1]
        
        # 종료점 찾기 (회복점)
        recovery_value = rolling_max.loc[max_dd_idx]
        recovery_candidates = cumulative_returns[max_dd_idx:]
        recovery_points = recovery_candidates[recovery_candidates >= recovery_value]
        
        if len(recovery_points) > 0:
            max_dd_end = recovery_points.index[0]
        else:
            max_dd_end = cumulative_returns.index[-1]  # 아직 회복 안됨
        
        # 기간 계산
        if max_dd_start and max_dd_end:
            max_dd_duration = (max_dd_end - max_dd_start).days
    
    # Average Drawdown
    drawdown_periods = []
    current_dd_start = None
    
    for i, (date, dd) in enumerate(drawdown.items()):
        if dd < 0 and current_dd_start is None:
            current_dd_start = i
        elif dd >= 0 and current_dd_start is not None:
            dd_period = drawdown.iloc[current_dd_start:i]
            if len(dd_period) > 0:
                drawdown_periods.append({
                    'max_dd': dd_period.min(),
                    'duration': len(dd_period),
                    'start_date': dd_period.index[0],
                    'end_date': dd_period.index[-1]
                })
            current_dd_start = None
    
    # 마지막 드로우다운이 끝나지 않은 경우
    if current_dd_start is not None:
        dd_period = drawdown.iloc[current_dd_start:]
        if len(dd_period) > 0:
            drawdown_periods.append({
                'max_dd': dd_period.min(),
                'duration': len(dd_period),
                'start_date': dd_period.index[0],
                'end_date': dd_period.index[-1]
            })
    
    # 평균 드로우다운
    if drawdown_periods:
        avg_drawdown = np.mean([dd['max_dd'] for dd in drawdown_periods])
        avg_dd_duration = np.mean([dd['duration'] for dd in drawdown_periods])
    else:
        avg_drawdown = 0.0
        avg_dd_duration = 0.0
    
    # Calmar Ratio (CAGR / |Max Drawdown|)
    cagr = calculate_returns_metrics(returns).get('cagr', 0)
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else np.inf
    
    return {
        'max_drawdown': max_drawdown,
        'max_dd_start_date': max_dd_start,
        'max_dd_end_date': max_dd_end,
        'max_dd_duration_days': max_dd_duration,
        'avg_drawdown': avg_drawdown,
        'avg_dd_duration_days': avg_dd_duration,
        'calmar_ratio': calmar_ratio,
        'drawdown_series': drawdown,
        'drawdown_periods': drawdown_periods,
        'n_drawdown_periods': len(drawdown_periods)
    }


def calculate_tail_risk_metrics(returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
    """꼬리 위험 지표를 계산합니다.
    
    Args:
        returns: 일일 수익률 시리즈
        confidence_levels: 신뢰수준 리스트
        
    Returns:
        꼬리 위험 지표 딕셔너리
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    metrics = {}
    
    for confidence in confidence_levels:
        alpha = 1 - confidence
        
        # Value at Risk (VaR)
        var = returns.quantile(alpha)
        metrics[f'var_{int(confidence*100)}'] = var
        
        # Conditional Value at Risk (CVaR) / Expected Shortfall
        cvar = returns[returns <= var].mean()
        metrics[f'cvar_{int(confidence*100)}'] = cvar
        
        # 연율화
        metrics[f'var_{int(confidence*100)}_annual'] = var * np.sqrt(252)
        metrics[f'cvar_{int(confidence*100)}_annual'] = cvar * np.sqrt(252)
    
    # Skewness and Kurtosis
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    metrics['excess_kurtosis'] = returns.kurtosis() - 3
    
    # Tail Ratio (95% quantile / 5% quantile)
    q95 = returns.quantile(0.95)
    q5 = returns.quantile(0.05)
    if q5 != 0:
        metrics['tail_ratio'] = q95 / abs(q5)
    else:
        metrics['tail_ratio'] = np.inf
    
    return metrics


def calculate_consistency_metrics(returns: pd.Series) -> Dict[str, float]:
    """일관성 지표를 계산합니다.
    
    Args:
        returns: 일일 수익률 시리즈
        
    Returns:
        일관성 지표 딕셔너리
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    # Win Rate (양의 수익률 비율)
    win_rate = (returns > 0).mean()
    
    # Best/Worst Day
    best_day = returns.max()
    worst_day = returns.min()
    
    # Profit Factor (총 이익 / 총 손실)
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) > 0 and negative_returns.sum() != 0:
        profit_factor = positive_returns.sum() / abs(negative_returns.sum())
    else:
        profit_factor = np.inf
    
    # Average Win/Loss
    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.0
    avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0.0
    
    # Win/Loss Ratio
    if avg_loss != 0:
        win_loss_ratio = abs(avg_win / avg_loss)
    else:
        win_loss_ratio = np.inf
    
    # Monthly returns analysis
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_win_rate = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0.0
    
    return {
        'win_rate': win_rate,
        'monthly_win_rate': monthly_win_rate,
        'best_day': best_day,
        'worst_day': worst_day,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'n_positive_days': len(positive_returns),
        'n_negative_days': len(negative_returns),
        'n_neutral_days': len(returns) - len(positive_returns) - len(negative_returns)
    }


def calculate_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    benchmark_returns: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """종합 성과 지표를 계산합니다.
    
    Args:
        returns: 일일 수익률 시리즈
        risk_free_rate: 무위험 수익률 (연율)
        benchmark_returns: 벤치마크 수익률 시리즈
        
    Returns:
        종합 성과 지표 딕셔너리
    """
    logger.debug(f"성과 지표 계산 시작: {len(returns)} 일")
    
    all_metrics = {}
    
    # 수익률 지표
    all_metrics.update(calculate_returns_metrics(returns))
    
    # 위험조정 지표
    all_metrics.update(calculate_risk_adjusted_metrics(returns, risk_free_rate))
    
    # 드로우다운 지표
    all_metrics.update(calculate_drawdown_metrics(returns))
    
    # 꼬리 위험 지표
    all_metrics.update(calculate_tail_risk_metrics(returns))
    
    # 일관성 지표
    all_metrics.update(calculate_consistency_metrics(returns))
    
    # 벤치마크 대비 지표
    if benchmark_returns is not None:
        benchmark_metrics = calculate_benchmark_metrics(returns, benchmark_returns)
        all_metrics.update(benchmark_metrics)
    
    logger.success(f"✅ 성과 지표 계산 완료: {len(all_metrics)} 지표")
    return all_metrics


def calculate_benchmark_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> Dict[str, float]:
    """벤치마크 대비 지표를 계산합니다.
    
    Args:
        returns: 포트폴리오 수익률
        benchmark_returns: 벤치마크 수익률
        
    Returns:
        벤치마크 대비 지표 딕셔너리
    """
    # 데이터 정렬
    aligned_data = pd.DataFrame({
        'portfolio': returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned_data) == 0:
        return {}
    
    port_returns = aligned_data['portfolio']
    bench_returns = aligned_data['benchmark']
    
    # 초과 수익률
    excess_returns = port_returns - bench_returns
    
    # 베타
    covariance = np.cov(port_returns, bench_returns)[0, 1]
    benchmark_variance = bench_returns.var()
    
    if benchmark_variance > 0:
        beta = covariance / benchmark_variance
    else:
        beta = 0.0
    
    # 알파 (연율화)
    portfolio_annual = port_returns.mean() * 252
    benchmark_annual = bench_returns.mean() * 252
    alpha = portfolio_annual - (beta * benchmark_annual)
    
    # Tracking Error
    tracking_error = excess_returns.std() * np.sqrt(252)
    
    # Information Ratio
    if tracking_error > 0:
        information_ratio = (excess_returns.mean() * 252) / tracking_error
    else:
        information_ratio = 0.0
    
    # Up/Down Capture Ratios
    up_benchmark = bench_returns[bench_returns > 0]
    down_benchmark = bench_returns[bench_returns < 0]
    
    if len(up_benchmark) > 0:
        up_portfolio = port_returns.loc[up_benchmark.index]
        up_capture = (up_portfolio.mean() / up_benchmark.mean()) if up_benchmark.mean() > 0 else 0.0
    else:
        up_capture = 0.0
    
    if len(down_benchmark) > 0:
        down_portfolio = port_returns.loc[down_benchmark.index]
        down_capture = (down_portfolio.mean() / down_benchmark.mean()) if down_benchmark.mean() < 0 else 0.0
    else:
        down_capture = 0.0
    
    return {
        'alpha': alpha,
        'beta': beta,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'up_capture': up_capture,
        'down_capture': down_capture,
        'excess_return_annual': excess_returns.mean() * 252
    }


def create_performance_summary(metrics: Dict[str, Any]) -> pd.DataFrame:
    """성과 지표 요약 테이블을 생성합니다.
    
    Args:
        metrics: 성과 지표 딕셔너리
        
    Returns:
        요약 테이블 데이터프레임
    """
    summary_data = []
    
    # 주요 지표 선택
    key_metrics = {
        'Total Return': ('total_return', '{:.2%}'),
        'CAGR': ('cagr', '{:.2%}'),
        'Volatility': ('volatility', '{:.2%}'),
        'Sharpe Ratio': ('sharpe_ratio', '{:.3f}'),
        'Sortino Ratio': ('sortino_ratio', '{:.3f}'),
        'Max Drawdown': ('max_drawdown', '{:.2%}'),
        'Calmar Ratio': ('calmar_ratio', '{:.3f}'),
        'VaR (95%)': ('var_95', '{:.2%}'),
        'CVaR (95%)': ('cvar_95', '{:.2%}'),
        'Win Rate': ('win_rate', '{:.2%}'),
        'Profit Factor': ('profit_factor', '{:.3f}'),
        'Best Day': ('best_day', '{:.2%}'),
        'Worst Day': ('worst_day', '{:.2%}')
    }
    
    for label, (key, format_str) in key_metrics.items():
        if key in metrics:
            value = metrics[key]
            if pd.isna(value) or np.isinf(value):
                formatted_value = 'N/A'
            else:
                try:
                    formatted_value = format_str.format(value)
                except (ValueError, TypeError):
                    formatted_value = str(value)
            
            summary_data.append({
                'Metric': label,
                'Value': formatted_value
            })
    
    return pd.DataFrame(summary_data)


def plot_performance_charts(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """성과 차트를 생성합니다.
    
    Args:
        returns: 포트폴리오 수익률
        benchmark_returns: 벤치마크 수익률
        save_path: 저장 경로
        
    Returns:
        차트 정보 딕셔너리
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 스타일 설정
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. 누적 수익률
        cumulative = (1 + returns).cumprod()
        axes[0, 0].plot(cumulative.index, cumulative.values, label='Portfolio', linewidth=2)
        
        if benchmark_returns is not None:
            bench_cumulative = (1 + benchmark_returns).cumprod()
            axes[0, 0].plot(bench_cumulative.index, bench_cumulative.values, 
                          label='Benchmark', linewidth=2, alpha=0.7)
        
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 드로우다운
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, 
                               alpha=0.3, color='red', label='Drawdown')
        axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 수익률 히스토그램
        axes[1, 0].hist(returns.values, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', 
                          label=f'Mean: {returns.mean():.4f}')
        axes[1, 0].set_title('Return Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 롤링 샤프 비율
        rolling_sharpe = returns.rolling(window=252).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252))
        )
        
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Rolling Sharpe Ratio (1-Year)')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 성과 차트 저장: {save_path}")
        
        return {
            'figure': fig,
            'axes': axes,
            'cumulative_returns': cumulative,
            'drawdown': drawdown,
            'rolling_sharpe': rolling_sharpe
        }
        
    except ImportError:
        logger.warning("matplotlib이 설치되지 않아 차트를 생성할 수 없습니다.")
        return {}
    except Exception as e:
        logger.error(f"차트 생성 실패: {e}")
        return {}


# CLI 직접 실행용
if __name__ == "__main__":
    # 테스트용 샘플 데이터 생성
    np.random.seed(42)
    
    # 임의의 수익률 시리즈 생성
    n_days = 252
    daily_returns = np.random.normal(0.0005, 0.015, n_days)  # 평균 0.05%, 표준편차 1.5%
    
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    returns_series = pd.Series(daily_returns, index=dates)
    
    print("🧪 성과 지표 계산 테스트 시작")
    
    # 성과 지표 계산
    metrics = calculate_performance_metrics(returns_series)
    
    # 요약 테이블 생성
    summary = create_performance_summary(metrics)
    print("\n📊 성과 지표 요약:")
    print(summary.to_string(index=False))
    
    # 주요 지표 출력
    print(f"\n🎯 주요 지표:")
    print(f"CAGR: {metrics.get('cagr', 0):.2%}")
    print(f"변동성: {metrics.get('volatility', 0):.2%}")
    print(f"샤프 비율: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"최대 낙폭: {metrics.get('max_drawdown', 0):.2%}")
    print(f"칼마 비율: {metrics.get('calmar_ratio', 0):.3f}")
    
    print("\n✅ 테스트 완료")