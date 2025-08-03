"""
백테스트 엔진 모듈

vectorbt를 기반으로 한 고성능 백테스트 실행 엔진입니다.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
import pickle
import warnings

import pandas as pd
import numpy as np
from loguru import logger

# vectorbt import (옵션)
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
    logger.info("vectorbt 사용 가능")
except ImportError:
    VBT_AVAILABLE = False
    logger.warning("vectorbt가 설치되지 않았습니다. 기본 백테스트 엔진을 사용합니다.")

from ..config import settings
from ..utils.paths import get_backtest_result_path, ensure_dir
from .metrics import calculate_performance_metrics


class BacktestEngine:
    """백테스트 엔진 클래스"""
    
    def __init__(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0002,
        min_trade_size: float = 100.0
    ):
        """
        Args:
            prices: 가격 데이터프레임 (인덱스: 날짜, 컬럼: 종목)
            initial_capital: 초기 자본
            transaction_cost: 거래 비용 (비율)
            slippage: 슬리피지 (비율)
            min_trade_size: 최소 거래 금액
        """
        self.prices = prices.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.min_trade_size = min_trade_size
        
        # 수익률 계산
        self.returns = self.prices.pct_change().fillna(0)
        
        logger.info(f"백테스트 엔진 초기화: {prices.shape}")
    
    def run_with_vectorbt(
        self,
        weights: pd.DataFrame,
        rebalance_freq: str = '1M',
        **kwargs
    ) -> Dict[str, Any]:
        """vectorbt를 사용한 백테스트 실행
        
        Args:
            weights: 가중치 데이터프레임
            rebalance_freq: 리밸런싱 주기
            **kwargs: 추가 매개변수
            
        Returns:
            백테스트 결과 딕셔너리
        """
        if not VBT_AVAILABLE:
            raise ImportError("vectorbt가 설치되지 않았습니다.")
        
        logger.info("🚀 vectorbt 백테스트 시작")
        
        try:
            # vectorbt 포트폴리오 생성
            portfolio = vbt.Portfolio.from_weights(
                close=self.prices,
                weights=weights,
                cash_sharing=True,
                init_cash=self.initial_capital,
                fees=self.transaction_cost,
                slippage=self.slippage,
                freq=rebalance_freq,
                **kwargs
            )
            
            # 성과 지표 계산
            total_return = portfolio.total_return()
            sharpe_ratio = portfolio.sharpe_ratio()
            max_drawdown = portfolio.max_drawdown()
            
            # 포트폴리오 가치
            portfolio_value = portfolio.value()
            
            # 거래 내역
            trades = portfolio.trades.records_readable if hasattr(portfolio, 'trades') else None
            
            results = {
                'engine': 'vectorbt',
                'portfolio': portfolio,
                'portfolio_value': portfolio_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trades': trades,
                'final_value': portfolio_value.iloc[-1],
                'success': True
            }
            
            logger.success("✅ vectorbt 백테스트 완료")
            return results
            
        except Exception as e:
            logger.error(f"❌ vectorbt 백테스트 실패: {e}")
            return {'success': False, 'error': str(e), 'engine': 'vectorbt'}
    
    def run_basic(
        self,
        weights: pd.DataFrame,
        rebalance_freq: str = '1M'
    ) -> Dict[str, Any]:
        """기본 백테스트 엔진으로 실행
        
        Args:
            weights: 가중치 데이터프레임
            rebalance_freq: 리밸런싱 주기
            
        Returns:
            백테스트 결과 딕셔너리
        """
        logger.info("🚀 기본 백테스트 엔진 시작")
        
        try:
            # 리밸런싱 날짜 결정
            rebalance_dates = self._get_rebalance_dates(rebalance_freq)
            
            # 포트폴리오 시뮬레이션
            portfolio_values = []
            cash_values = []
            positions = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
            
            current_cash = self.initial_capital
            current_positions = pd.Series(0.0, index=self.prices.columns)
            
            for i, date in enumerate(self.prices.index):
                # 리밸런싱 체크
                if date in rebalance_dates or i == 0:
                    current_cash, current_positions = self._rebalance(
                        date, weights.loc[date], current_cash, current_positions
                    )
                
                # 포지션 업데이트 (가격 변화 반영)
                if i > 0:
                    price_changes = self.prices.loc[date] / self.prices.iloc[i-1]
                    current_positions *= price_changes
                
                # 포트폴리오 가치 계산
                portfolio_value = current_cash + (current_positions * self.prices.loc[date]).sum()
                
                portfolio_values.append(portfolio_value)
                cash_values.append(current_cash)
                positions.loc[date] = current_positions
            
            # 결과 정리
            portfolio_series = pd.Series(portfolio_values, index=self.prices.index)
            cash_series = pd.Series(cash_values, index=self.prices.index)
            
            # 수익률 계산
            portfolio_returns = portfolio_series.pct_change().fillna(0)
            
            results = {
                'engine': 'basic',
                'portfolio_value': portfolio_series,
                'cash_value': cash_series,
                'positions': positions,
                'portfolio_returns': portfolio_returns,
                'total_return': (portfolio_series.iloc[-1] / self.initial_capital) - 1,
                'final_value': portfolio_series.iloc[-1],
                'success': True
            }
            
            logger.success("✅ 기본 백테스트 완료")
            return results
            
        except Exception as e:
            logger.error(f"❌ 기본 백테스트 실패: {e}")
            return {'success': False, 'error': str(e), 'engine': 'basic'}
    
    def _get_rebalance_dates(self, freq: str) -> List[pd.Timestamp]:
        """리밸런싱 날짜를 생성합니다.
        
        Args:
            freq: 리밸런싱 주기 ('1D', '1W', '1M', '1Q' 등)
            
        Returns:
            리밸런싱 날짜 리스트
        """
        if freq == '1D' or freq == 'daily':
            return self.prices.index.tolist()
        
        # pandas의 resample을 이용한 주기 생성
        dummy_series = pd.Series(1, index=self.prices.index)
        resampled = dummy_series.resample(freq).first()
        
        return resampled.index.tolist()
    
    def _rebalance(
        self,
        date: pd.Timestamp,
        target_weights: pd.Series,
        current_cash: float,
        current_positions: pd.Series
    ) -> tuple:
        """리밸런싱을 실행합니다.
        
        Args:
            date: 리밸런싱 날짜
            target_weights: 목표 가중치
            current_cash: 현재 현금
            current_positions: 현재 포지션
            
        Returns:
            (업데이트된 현금, 업데이트된 포지션)
        """
        current_prices = self.prices.loc[date]
        
        # 현재 포트폴리오 가치
        current_portfolio_value = current_cash + (current_positions * current_prices).sum()
        
        # 목표 포지션 계산
        target_values = target_weights * current_portfolio_value
        target_positions = target_values / current_prices
        
        # 거래 실행
        trades = target_positions - current_positions
        
        for asset in trades.index:
            trade_size = trades[asset]
            
            if abs(trade_size * current_prices[asset]) < self.min_trade_size:
                continue
            
            # 거래 비용 및 슬리피지 적용
            trade_value = abs(trade_size * current_prices[asset])
            cost = trade_value * (self.transaction_cost + self.slippage)
            
            if trade_size > 0:  # 매수
                if current_cash >= trade_value + cost:
                    current_cash -= (trade_value + cost)
                    current_positions[asset] += trade_size
            else:  # 매도
                current_cash += (trade_value - cost)
                current_positions[asset] += trade_size
        
        return current_cash, current_positions


def run_backtest(
    strategy_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    rebalance_freq: str = '1M',
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0002,
    save_results: bool = True,
    **strategy_kwargs
) -> Dict[str, Any]:
    """백테스트를 실행하는 메인 함수
    
    Args:
        strategy_name: 전략 이름 ('equal_weight', 'vol_parity' 등)
        start_date: 시작 날짜
        end_date: 종료 날짜
        rebalance_freq: 리밸런싱 주기
        initial_capital: 초기 자본
        transaction_cost: 거래 비용
        slippage: 슬리피지
        save_results: 결과 저장 여부
        **strategy_kwargs: 전략별 매개변수
        
    Returns:
        백테스트 결과 딕셔너리
    """
    logger.info(f"🚀 백테스트 시작: {strategy_name}")
    
    # 피처 데이터 로드
    try:
        from ..features.ta_factors import load_features
        features_data = load_features()
        
        # 가격 데이터 추출
        if isinstance(features_data.index, pd.MultiIndex):
            prices = features_data['close'].unstack(level=1)
        else:
            raise ValueError("MultiIndex 데이터가 필요합니다.")
            
    except FileNotFoundError:
        logger.error("피처 데이터를 찾을 수 없습니다. 먼저 'make-features' 명령을 실행하세요.")
        return {'success': False, 'error': 'Feature data not found'}
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return {'success': False, 'error': str(e)}
    
    # 날짜 범위 필터링
    if start_date:
        prices = prices[prices.index >= start_date]
    if end_date:
        prices = prices[prices.index <= end_date]
        
    if prices.empty:
        logger.error("지정된 날짜 범위에 데이터가 없습니다.")
        return {'success': False, 'error': 'No data in date range'}
    
    # 전략 모듈 동적 로드
    try:
        if strategy_name == 'equal_weight':
            from ..strategies.equal_weight import weights as strategy_weights
        elif strategy_name == 'vol_parity':
            from ..strategies.vol_parity import weights as strategy_weights
        else:
            raise ValueError(f"지원하지 않는 전략: {strategy_name}")
            
    except ImportError as e:
        logger.error(f"전략 모듈 로드 실패: {e}")
        return {'success': False, 'error': f'Strategy import failed: {e}'}
    
    # 전략 가중치 계산
    try:
        weights_df = strategy_weights(prices, **strategy_kwargs)
    except Exception as e:
        logger.error(f"전략 가중치 계산 실패: {e}")
        return {'success': False, 'error': f'Weight calculation failed: {e}'}
    
    # 백테스트 엔진 초기화
    engine = BacktestEngine(
        prices=prices,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        slippage=slippage
    )
    
    # 백테스트 실행 (vectorbt 우선, 실패시 기본 엔진)
    if VBT_AVAILABLE:
        results = engine.run_with_vectorbt(weights_df, rebalance_freq)
        if not results.get('success', False):
            logger.warning("vectorbt 실패, 기본 엔진으로 전환")
            results = engine.run_basic(weights_df, rebalance_freq)
    else:
        results = engine.run_basic(weights_df, rebalance_freq)
    
    if not results.get('success', False):
        return results
    
    # 성과 지표 계산
    try:
        if 'portfolio_returns' in results:
            performance_metrics = calculate_performance_metrics(
                results['portfolio_returns']
            )
        else:
            # vectorbt 결과에서 수익률 추출
            portfolio_value = results['portfolio_value']
            portfolio_returns = portfolio_value.pct_change().fillna(0)
            performance_metrics = calculate_performance_metrics(portfolio_returns)
            results['portfolio_returns'] = portfolio_returns
        
        results.update(performance_metrics)
        
    except Exception as e:
        logger.warning(f"성과 지표 계산 실패: {e}")
    
    # 결과 메타데이터 추가
    results.update({
        'backtest_id': datetime.now().strftime('%Y%m%dT%H%M%S'),
        'strategy_name': strategy_name,
        'start_date': prices.index[0],
        'end_date': prices.index[-1],
        'total_days': len(prices),
        'n_assets': prices.shape[1],
        'rebalance_freq': rebalance_freq,
        'initial_capital': initial_capital,
        'transaction_cost': transaction_cost,
        'slippage': slippage,
        'strategy_kwargs': strategy_kwargs,
        'weights': weights_df
    })
    
    # 결과 저장
    if save_results:
        try:
            save_path = get_backtest_result_path(results['backtest_id'])
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"💾 백테스트 결과 저장: {save_path}")
            results['save_path'] = save_path
        except Exception as e:
            logger.warning(f"결과 저장 실패: {e}")
    
    logger.success(f"✅ 백테스트 완료: {strategy_name}")
    return results


def load_backtest_result(backtest_id: str) -> Dict[str, Any]:
    """저장된 백테스트 결과를 로드합니다.
    
    Args:
        backtest_id: 백테스트 ID
        
    Returns:
        백테스트 결과 딕셔너리
    """
    save_path = get_backtest_result_path(backtest_id)
    
    if not save_path.exists():
        raise FileNotFoundError(f"백테스트 결과를 찾을 수 없습니다: {save_path}")
    
    with open(save_path, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"📂 백테스트 결과 로드: {backtest_id}")
    return results


def list_backtest_results() -> List[Dict[str, Any]]:
    """저장된 백테스트 결과 목록을 반환합니다.
    
    Returns:
        백테스트 결과 목록
    """
    results_dir = get_backtest_result_path()
    
    if not results_dir.exists():
        return []
    
    result_files = list(results_dir.glob("*.pkl"))
    results_list = []
    
    for file_path in result_files:
        try:
            with open(file_path, 'rb') as f:
                result = pickle.load(f)
            
            summary = {
                'backtest_id': result.get('backtest_id', file_path.stem),
                'strategy_name': result.get('strategy_name', 'Unknown'),
                'start_date': result.get('start_date'),
                'end_date': result.get('end_date'),
                'total_return': result.get('total_return'),
                'sharpe_ratio': result.get('sharpe_ratio'),
                'max_drawdown': result.get('max_drawdown'),
                'file_path': file_path,
                'created_time': datetime.fromtimestamp(file_path.stat().st_mtime)
            }
            
            results_list.append(summary)
            
        except Exception as e:
            logger.warning(f"백테스트 결과 로드 실패: {file_path} - {e}")
            continue
    
    # 생성 시간순 정렬 (최신 순)
    results_list.sort(key=lambda x: x['created_time'], reverse=True)
    
    return results_list


def compare_strategies(
    strategy_results: List[Dict[str, Any]],
    metrics: List[str] = None
) -> pd.DataFrame:
    """여러 전략의 성과를 비교합니다.
    
    Args:
        strategy_results: 전략 결과 리스트
        metrics: 비교할 지표 리스트
        
    Returns:
        비교 결과 데이터프레임
    """
    if metrics is None:
        metrics = [
            'total_return', 'annualized_return', 'volatility', 
            'sharpe_ratio', 'max_drawdown', 'calmar_ratio'
        ]
    
    comparison_data = []
    
    for result in strategy_results:
        strategy_metrics = {
            'strategy': result.get('strategy_name', 'Unknown'),
            'backtest_id': result.get('backtest_id', 'Unknown')
        }
        
        for metric in metrics:
            strategy_metrics[metric] = result.get(metric, np.nan)
        
        comparison_data.append(strategy_metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('strategy')
    
    return comparison_df


# CLI 직접 실행용
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python -m src.backtest.engine STRATEGY_NAME")
        print("예시: python -m src.backtest.engine equal_weight")
        sys.exit(1)
    
    strategy = sys.argv[1]
    
    # 테스트 실행
    result = run_backtest(
        strategy_name=strategy,
        start_date='2021-01-01',
        end_date='2024-12-31'
    )
    
    if result.get('success', False):
        print(f"\n✅ {strategy} 백테스트 완료:")
        print(f"총 수익률: {result.get('total_return', 0):.2%}")
        print(f"연간 수익률: {result.get('annualized_return', 0):.2%}")
        print(f"샤프 비율: {result.get('sharpe_ratio', 0):.2f}")
        print(f"최대 낙폭: {result.get('max_drawdown', 0):.2%}")
    else:
        print(f"❌ 백테스트 실패: {result.get('error', 'Unknown error')}")
        sys.exit(1)