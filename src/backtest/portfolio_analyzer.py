"""
통합 포트폴리오 분석기
사용자 친화적 인터페이스로 백테스트부터 리포트까지 자동 생성
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import pickle
import json

# 시각화
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import seaborn as sns

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# 로깅
from loguru import logger

warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    """
    통합 포트폴리오 분석기
    - 종목/비중/전략 설정
    - 백테스트 실행
    - 상세 지표 계산
    - 리포트 자동 생성
    """
    
    def __init__(self, data_path: str = "data/silver/features.parquet"):
        """
        Args:
            data_path: 피처 데이터 파일 경로
        """
        self.data_path = Path(data_path)
        self.df = None
        self.prices = None
        self.returns = None
        
        # 결과 저장
        self.results = {}
        self.portfolio_specs = {}
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        logger.info("🚀 PortfolioAnalyzer 초기화 완료")
    
    def load_data(self, start_date: str = "2020-01-01", end_date: str = "2024-12-31"):
        """데이터 로드 및 전처리"""
        logger.info(f"📊 데이터 로드: {start_date} ~ {end_date}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"데이터 파일이 없습니다: {self.data_path}")
        
        # 데이터 로드
        self.df = pd.read_parquet(self.data_path).reset_index()
        
        # 날짜 필터링
        self.df = self.df[
            (self.df['date'] >= start_date) & 
            (self.df['date'] <= end_date)
        ].copy()
        
        logger.success(f"✅ 데이터 로드 완료: {len(self.df)} 행")
        return self
    
    def set_portfolio(self, 
                     tickers: List[str], 
                     weights: Optional[List[float]] = None,
                     name: str = "Custom Portfolio"):
        """
        포트폴리오 구성 설정
        
        Args:
            tickers: 종목 리스트
            weights: 비중 리스트 (None이면 동일가중)
            name: 포트폴리오 이름
        """
        if weights is None:
            weights = [1.0 / len(tickers)] * len(tickers)
        
        if len(tickers) != len(weights):
            raise ValueError("종목 수와 비중 수가 일치하지 않습니다")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("비중의 합이 1이 아닙니다")
        
        # 데이터 필터링
        df_filtered = self.df[self.df['ticker'].isin(tickers)].copy()
        
        # 가격 데이터 피벗
        self.prices = df_filtered.pivot(index='date', columns='ticker', values='close')
        self.prices = self.prices[tickers].dropna()  # 순서 보장 및 결측치 제거
        
        # 수익률 계산
        self.returns = self.prices.pct_change().dropna()
        
        # 포트폴리오 설정 저장
        self.portfolio_specs = {
            'name': name,
            'tickers': tickers,
            'weights': weights,
            'start_date': self.prices.index.min().strftime('%Y-%m-%d'),
            'end_date': self.prices.index.max().strftime('%Y-%m-%d'),
            'trading_days': len(self.prices)
        }
        
        logger.info(f"🎯 포트폴리오 설정: {name}")
        logger.info(f"   종목: {tickers}")
        logger.info(f"   비중: {[f'{w:.1%}' for w in weights]}")
        logger.info(f"   기간: {self.portfolio_specs['start_date']} ~ {self.portfolio_specs['end_date']}")
        
        return self
    
    def run_strategy_comparison(self, 
                               rebalance_periods: List[str] = ['1M', '3M', '6M', '1Y'],
                               strategies: List[str] = ['equal_weight', 'vol_parity']):
        """
        여러 전략과 리밸런싱 주기 비교
        
        Args:
            rebalance_periods: 리밸런싱 주기 리스트
            strategies: 전략 리스트
        """
        logger.info(f"🧪 전략 비교 시작: {strategies} × {rebalance_periods}")
        
        self.results = {}
        
        for strategy in strategies:
            self.results[strategy] = {}
            
            for period in rebalance_periods:
                logger.info(f"   실행 중: {strategy} - {period}")
                
                if strategy == 'equal_weight':
                    result = self._run_equal_weight(period)
                elif strategy == 'vol_parity':
                    result = self._run_vol_parity(period)
                elif strategy == 'custom':
                    result = self._run_custom_weights(period)
                else:
                    raise ValueError(f"지원하지 않는 전략: {strategy}")
                
                self.results[strategy][period] = result
        
        logger.success("✅ 전략 비교 완료")
        return self
    
    def _run_equal_weight(self, rebalance_period: str) -> Dict:
        """동일가중 전략 실행"""
        weights = np.array(self.portfolio_specs['weights'])
        return self._backtest_with_rebalancing(weights, rebalance_period, 'Equal Weight')
    
    def _run_vol_parity(self, rebalance_period: str, vol_window: int = 20) -> Dict:
        """변동성 패리티 전략 실행"""
        # 롤링 변동성 계산
        rolling_vol = self.returns.rolling(vol_window).std()
        inverse_vol = 1 / rolling_vol
        dynamic_weights = inverse_vol.div(inverse_vol.sum(axis=1), axis=0).fillna(0)
        
        return self._backtest_with_dynamic_weights(dynamic_weights, rebalance_period, 'Volatility Parity')
    
    def _run_custom_weights(self, rebalance_period: str) -> Dict:
        """사용자 정의 비중 전략"""
        weights = np.array(self.portfolio_specs['weights'])
        return self._backtest_with_rebalancing(weights, rebalance_period, 'Custom Weights')
    
    def _backtest_with_rebalancing(self, weights: np.ndarray, rebalance_period: str, strategy_name: str) -> Dict:
        """정적 가중치로 리밸런싱 백테스트"""
        
        # 리밸런싱 날짜 생성
        rebalance_dates = self._get_rebalance_dates(rebalance_period)
        
        # 포트폴리오 시뮬레이션
        portfolio_values = []
        portfolio_returns = []
        rebalancing_costs = []
        current_weights = weights.copy()
        
        initial_value = 100000  # 초기 자본
        portfolio_value = initial_value
        
        prev_date = None
        transaction_cost = 0.001  # 0.1% 거래비용
        
        for date in self.returns.index:
            daily_returns = self.returns.loc[date].values
            
            # 자연적 가중치 변화 (수익률 반영)
            if prev_date is not None:
                current_weights = current_weights * (1 + daily_returns)
                current_weights = current_weights / current_weights.sum()
            
            # 리밸런싱 체크
            rebalancing_cost = 0
            if date in rebalance_dates:
                # 리밸런싱 비용 계산
                weight_diff = np.abs(current_weights - weights).sum()
                rebalancing_cost = portfolio_value * weight_diff * transaction_cost
                current_weights = weights.copy()
            
            # 포트폴리오 가치 업데이트
            daily_portfolio_return = np.sum(current_weights * daily_returns)
            portfolio_value = portfolio_value * (1 + daily_portfolio_return) - rebalancing_cost
            
            portfolio_values.append(portfolio_value)
            portfolio_returns.append(daily_portfolio_return)
            rebalancing_costs.append(rebalancing_cost)
            
            prev_date = date
        
        # 결과 정리
        portfolio_series = pd.Series(portfolio_values, index=self.returns.index)
        returns_series = pd.Series(portfolio_returns, index=self.returns.index)
        
        return {
            'strategy_name': strategy_name,
            'rebalance_period': rebalance_period,
            'portfolio_values': portfolio_series,
            'portfolio_returns': returns_series,
            'rebalancing_costs': pd.Series(rebalancing_costs, index=self.returns.index),
            'rebalance_dates': rebalance_dates,
            'total_rebalancing_cost': sum(rebalancing_costs),
            'rebalancing_count': len([c for c in rebalancing_costs if c > 0]),
            'final_value': portfolio_value,
            'weights': weights,
            'metrics': self._calculate_comprehensive_metrics(returns_series, portfolio_series)
        }
    
    def _backtest_with_dynamic_weights(self, dynamic_weights: pd.DataFrame, rebalance_period: str, strategy_name: str) -> Dict:
        """동적 가중치로 백테스트"""
        
        rebalance_dates = self._get_rebalance_dates(rebalance_period)
        
        portfolio_values = []
        portfolio_returns = []
        rebalancing_costs = []
        weight_history = []
        
        initial_value = 100000
        portfolio_value = initial_value
        current_weights = None
        transaction_cost = 0.001
        
        for i, date in enumerate(self.returns.index):
            daily_returns = self.returns.loc[date].values
            target_weights = dynamic_weights.loc[date].values
            
            # 첫 날이거나 리밸런싱 날짜
            rebalancing_cost = 0
            if current_weights is None or date in rebalance_dates:
                if current_weights is not None:
                    weight_diff = np.abs(current_weights - target_weights).sum()
                    rebalancing_cost = portfolio_value * weight_diff * transaction_cost
                current_weights = target_weights.copy()
            else:
                # 자연적 가중치 변화
                current_weights = current_weights * (1 + daily_returns)
                current_weights = current_weights / current_weights.sum()
            
            # 포트폴리오 수익률 계산
            daily_portfolio_return = np.sum(current_weights * daily_returns)
            portfolio_value = portfolio_value * (1 + daily_portfolio_return) - rebalancing_cost
            
            portfolio_values.append(portfolio_value)
            portfolio_returns.append(daily_portfolio_return)
            rebalancing_costs.append(rebalancing_cost)
            weight_history.append(current_weights.copy())
        
        # 결과 정리
        portfolio_series = pd.Series(portfolio_values, index=self.returns.index)
        returns_series = pd.Series(portfolio_returns, index=self.returns.index)
        weights_df = pd.DataFrame(weight_history, index=self.returns.index, columns=self.prices.columns)
        
        return {
            'strategy_name': strategy_name,
            'rebalance_period': rebalance_period,
            'portfolio_values': portfolio_series,
            'portfolio_returns': returns_series,
            'rebalancing_costs': pd.Series(rebalancing_costs, index=self.returns.index),
            'rebalance_dates': rebalance_dates,
            'total_rebalancing_cost': sum(rebalancing_costs),
            'rebalancing_count': len([c for c in rebalancing_costs if c > 0]),
            'final_value': portfolio_value,
            'dynamic_weights': weights_df,
            'metrics': self._calculate_comprehensive_metrics(returns_series, portfolio_series)
        }
    
    def _get_rebalance_dates(self, period: str) -> List[pd.Timestamp]:
        """리밸런싱 날짜 생성"""
        if period == '1D':
            return list(self.returns.index)
        
        # 첫째 날 포함
        dates = [self.returns.index[0]]
        
        # 주기적 리밸런싱 날짜 추가
        freq_map = {'1W': 'W', '1M': 'MS', '3M': '3MS', '6M': '6MS', '1Y': 'YS'}
        freq = freq_map.get(period, 'MS')
        
        # pandas의 date_range로 주기적 날짜 생성
        start = self.returns.index[0]
        end = self.returns.index[-1]
        
        period_dates = pd.date_range(start=start, end=end, freq=freq)
        
        # 실제 거래일에 맞춰 조정
        for date in period_dates:
            # 가장 가까운 다음 거래일 찾기
            future_dates = self.returns.index[self.returns.index >= date]
            if len(future_dates) > 0:
                actual_date = future_dates[0]
                if actual_date not in dates:
                    dates.append(actual_date)
        
        return sorted(dates)
    
    def _calculate_comprehensive_metrics(self, returns: pd.Series, portfolio_values: pd.Series) -> Dict:
        """종합적인 성과 지표 계산"""
        
        # 기본 통계
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 위험 조정 지표
        risk_free_rate = 0.02  # 2% 무위험 수익률
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 드로우다운 분석
        cumulative = portfolio_values / portfolio_values.iloc[0]
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 드로우다운 지속기간
        dd_duration = self._calculate_drawdown_duration(drawdown)
        
        # 소르티노 비율 (하방 변동성만 고려)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (returns.mean() * 252 - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # 칼마 비율
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR & CVaR (95% 신뢰도)
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # 베타 (시장 대비) - QQQ를 벤치마크로 사용
        if 'QQQ' in self.prices.columns:
            benchmark_returns = self.returns['QQQ']
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
            
            # 알파 (CAPM)
            alpha = returns.mean() * 252 - (risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate))
            
            # 정보 비율
            active_returns = returns - benchmark_returns
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
        else:
            beta = 1.0
            alpha = 0.0
            information_ratio = 0.0
        
        # 승률 & 손익비
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 변동성 지표
        upside_volatility = returns[returns > returns.mean()].std() * np.sqrt(252)
        downside_volatility = returns[returns < returns.mean()].std() * np.sqrt(252)
        
        # 월별 수익률 분석
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        positive_months = len(monthly_returns[monthly_returns > 0])
        total_months = len(monthly_returns)
        monthly_win_rate = positive_months / total_months if total_months > 0 else 0
        
        # 최대 연속 상승/하락일
        returns_sign = np.sign(returns)
        max_consecutive_wins = self._max_consecutive(returns_sign, 1)
        max_consecutive_losses = self._max_consecutive(returns_sign, -1)
        
        return {
            # 수익률 지표
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'best_day': returns.max(),
            'worst_day': returns.min(),
            
            # 위험 조정 지표  
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            
            # 드로우다운 지표
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': dd_duration['avg_duration'],
            'max_drawdown_duration': dd_duration['max_duration'],
            
            # 위험 지표
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_volatility': downside_volatility,
            'upside_volatility': upside_volatility,
            
            # 시장 지표
            'beta': beta,
            'alpha': alpha,
            
            # 일관성 지표
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'monthly_win_rate': monthly_win_rate,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            
            # 추가 통계
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'trading_days': len(returns),
            'positive_days': len(positive_returns),
            'negative_days': len(negative_returns)
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> Dict:
        """드로우다운 지속기간 계산"""
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                # 드로우다운 시작
                in_drawdown = True
                start_date = date
            elif dd >= 0 and in_drawdown:
                # 드로우다운 종료
                in_drawdown = False
                if start_date:
                    duration = (date - start_date).days
                    drawdown_periods.append(duration)
        
        # 마지막까지 드로우다운 중인 경우
        if in_drawdown and start_date:
            duration = (drawdown.index[-1] - start_date).days
            drawdown_periods.append(duration)
        
        return {
            'avg_duration': np.mean(drawdown_periods) if drawdown_periods else 0,
            'max_duration': max(drawdown_periods) if drawdown_periods else 0,
            'total_periods': len(drawdown_periods)
        }
    
    def _max_consecutive(self, series: pd.Series, value: int) -> int:
        """최대 연속 발생 횟수"""
        max_count = 0
        current_count = 0
        
        for v in series:
            if v == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def generate_comprehensive_report(self, output_dir: str = "reports") -> str:
        """통합 리포트 생성 (차트 + HTML + MD)"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 차트 생성 (저장만, plt.show() 안함)
        chart_path = self._generate_comprehensive_charts(output_path, timestamp)
        
        # 2. 고급 리포터로 HTML/MD 생성
        from ..reports.advanced_reporter import AdvancedReporter
        reporter = AdvancedReporter(self.results, self.portfolio_specs)
        
        html_path = reporter.generate_html_report(output_path, timestamp)
        md_path = reporter.generate_markdown_report(output_path, timestamp)
        
        logger.success(f"📊 종합 리포트 생성 완료:")
        logger.info(f"   차트: {chart_path}")
        logger.info(f"   HTML: {html_path}")
        logger.info(f"   MD: {md_path}")
        
        return str(html_path)
    
    def _generate_comprehensive_charts(self, output_path: Path, timestamp: str) -> Path:
        """종합 차트 생성 (plt.show() 제거)"""
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'포트폴리오 종합 분석 - {self.portfolio_specs["name"]}', fontsize=20, fontweight='bold')
        
        # 색상 팔레트
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
        
        # 차트별 데이터 준비
        strategies = list(self.results.keys())
        periods = list(self.results[strategies[0]].keys())
        
        # 1. 누적 수익률 비교 (전략별)
        ax1 = axes[0, 0]
        for i, strategy in enumerate(strategies):
            for period in periods:
                result = self.results[strategy][period]
                cumulative = result['portfolio_values'] / result['portfolio_values'].iloc[0]
                label = f"{strategy}_{period}"
                ax1.plot(cumulative.index, cumulative.values, 
                        label=label, linewidth=2, alpha=0.8)
        
        ax1.set_title('Cumulative Returns by Strategy')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 리밸런싱 주기별 성과 비교
        ax2 = axes[0, 1]
        strategy_name = strategies[0]  # 첫 번째 전략으로 비교
        cagrs = [self.results[strategy_name][period]['metrics']['cagr'] * 100 for period in periods]
        volatilities = [self.results[strategy_name][period]['metrics']['volatility'] * 100 for period in periods]
        
        ax2.bar(range(len(periods)), cagrs, alpha=0.7, color=colors[0], label='CAGR')
        ax2_twin = ax2.twinx()
        ax2_twin.bar([x + 0.4 for x in range(len(periods))], volatilities, 
                    alpha=0.7, color=colors[1], label='Volatility', width=0.4)
        
        ax2.set_title(f'CAGR vs Volatility by Rebalancing Period ({strategy_name})')
        ax2.set_xticks(range(len(periods)))
        ax2.set_xticklabels(periods)
        ax2.set_ylabel('CAGR (%)', color=colors[0])
        ax2_twin.set_ylabel('Volatility (%)', color=colors[1])
        
        # 3. 샤프 비율 히트맵
        ax3 = axes[0, 2]
        sharpe_matrix = []
        for strategy in strategies:
            row = [self.results[strategy][period]['metrics']['sharpe_ratio'] for period in periods]
            sharpe_matrix.append(row)
        
        im = ax3.imshow(sharpe_matrix, cmap='RdYlGn', aspect='auto')
        ax3.set_title('Sharpe Ratio Heatmap')
        ax3.set_xticks(range(len(periods)))
        ax3.set_xticklabels(periods)
        ax3.set_yticks(range(len(strategies)))
        ax3.set_yticklabels(strategies)
        
        # 값 표시
        for i in range(len(strategies)):
            for j in range(len(periods)):
                ax3.text(j, i, f'{sharpe_matrix[i][j]:.3f}', 
                        ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=ax3, shrink=0.8)
        
        # 4. 드로우다운 분석
        ax4 = axes[1, 0]
        for i, strategy in enumerate(strategies):
            period = periods[0]  # 첫 번째 기간으로 비교
            result = self.results[strategy][period]
            cumulative = result['portfolio_values'] / result['portfolio_values'].iloc[0]
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            ax4.fill_between(drawdown.index, drawdown.values, 0, 
                           alpha=0.6, color=colors[i], label=f'{strategy}_{period}')
        
        ax4.set_title('Drawdown Comparison')
        ax4.set_ylabel('Drawdown (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 리밸런싱 비용 분석
        ax5 = axes[1, 1]
        cost_data = []
        labels = []
        for strategy in strategies:
            for period in periods:
                result = self.results[strategy][period]
                total_cost = result['total_rebalancing_cost']
                cost_data.append(total_cost)
                labels.append(f"{strategy}\n{period}")
        
        bars = ax5.bar(range(len(cost_data)), cost_data, color=colors[:len(cost_data)])
        ax5.set_title('Total Rebalancing Costs')
        ax5.set_ylabel('Cost ($)')
        ax5.set_xticks(range(len(labels)))
        ax5.set_xticklabels(labels, rotation=45)
        
        # 값 표시
        for bar, cost in zip(bars, cost_data):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'${cost:.0f}', ha='center', va='bottom')
        
        # 6. 월별 수익률 분포
        ax6 = axes[1, 2]
        strategy_name = strategies[0]
        period_name = periods[0]
        result = self.results[strategy_name][period_name]
        monthly_returns = result['portfolio_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        ax6.hist(monthly_returns * 100, bins=20, alpha=0.7, color=colors[0], edgecolor='black')
        ax6.axvline(monthly_returns.mean() * 100, color='red', linestyle='--', 
                   label=f'Mean: {monthly_returns.mean()*100:.1f}%')
        ax6.set_title(f'Monthly Returns Distribution ({strategy_name}_{period_name})')
        ax6.set_xlabel('Monthly Return (%)')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. 변동성 분해 (상승/하락)
        ax7 = axes[2, 0]
        upside_vols = []
        downside_vols = []
        strategy_labels = []
        
        for strategy in strategies:
            period = periods[0]  # 첫 번째 기간
            metrics = self.results[strategy][period]['metrics']
            upside_vols.append(metrics['upside_volatility'] * 100)
            downside_vols.append(metrics['downside_volatility'] * 100)
            strategy_labels.append(strategy)
        
        x = np.arange(len(strategy_labels))
        width = 0.35
        
        ax7.bar(x - width/2, upside_vols, width, label='Upside Vol', color=colors[0], alpha=0.8)
        ax7.bar(x + width/2, downside_vols, width, label='Downside Vol', color=colors[1], alpha=0.8)
        
        ax7.set_title('Upside vs Downside Volatility')
        ax7.set_ylabel('Volatility (%)')
        ax7.set_xticks(x)
        ax7.set_xticklabels(strategy_labels)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. 위험 지표 레이더 차트
        ax8 = axes[2, 1]
        ax8.remove()  # 기존 axes 제거
        ax8 = fig.add_subplot(3, 3, 8, projection='polar')
        
        # 레이더 차트 데이터 준비
        strategy_name = strategies[0]
        period_name = periods[0]
        metrics = self.results[strategy_name][period_name]['metrics']
        
        categories = ['Sharpe', 'Sortino', 'Calmar', 'Win Rate', 'Profit Factor']
        values = [
            max(0, min(3, metrics['sharpe_ratio'])) / 3,  # 0-3 범위를 0-1로 정규화
            max(0, min(3, metrics['sortino_ratio'])) / 3,
            max(0, min(3, metrics['calmar_ratio'])) / 3,
            metrics['win_rate'],
            max(0, min(5, metrics['profit_factor'])) / 5  # 0-5 범위를 0-1로 정규화
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 원형으로 연결
        angles += angles[:1]
        
        ax8.plot(angles, values, 'o-', linewidth=2, color=colors[0])
        ax8.fill(angles, values, alpha=0.25, color=colors[0])
        ax8.set_xticks(angles[:-1])
        ax8.set_xticklabels(categories)
        ax8.set_ylim(0, 1)
        ax8.set_title(f'Risk Metrics Radar ({strategy_name}_{period_name})', pad=20)
        
        # 9. 개별 종목 기여도
        ax9 = axes[2, 2]
        if 'dynamic_weights' in self.results[strategies[0]][periods[0]]:
            # 동적 가중치 전략의 경우
            weights_df = self.results[strategies[0]][periods[0]]['dynamic_weights']
            avg_weights = weights_df.mean()
        else:
            # 정적 가중치 전략의 경우
            avg_weights = pd.Series(self.portfolio_specs['weights'], 
                                  index=self.portfolio_specs['tickers'])
        
        # 개별 종목 수익률
        individual_returns = {}
        for ticker in self.portfolio_specs['tickers']:
            if ticker in self.returns.columns:
                total_ret = (1 + self.returns[ticker]).prod() - 1
                individual_returns[ticker] = total_ret * 100
        
        # 기여도 계산 (가중치 × 개별수익률)
        contributions = []
        for ticker in avg_weights.index:
            if ticker in individual_returns:
                contrib = avg_weights[ticker] * individual_returns[ticker]
                contributions.append(contrib)
            else:
                contributions.append(0)
        
        bars = ax9.bar(avg_weights.index, contributions, color=colors[:len(avg_weights)])
        ax9.set_title('Individual Asset Contribution')
        ax9.set_ylabel('Contribution (%)')
        ax9.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, contrib in zip(bars, contributions):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{contrib:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 파일 저장 (plt.show() 안함)
        chart_path = output_path / f"comprehensive_analysis_{timestamp}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # 메모리 해제 및 창 열림 방지
        
        return chart_path


# 사용 편의를 위한 래퍼 함수
def quick_backtest(tickers: List[str], 
                   weights: Optional[List[float]] = None,
                   strategies: List[str] = ['equal_weight', 'vol_parity'],
                   rebalance_periods: List[str] = ['1M', '3M', '6M'],
                   start_date: str = "2022-01-01",
                   end_date: str = "2024-12-31",
                   portfolio_name: str = "Custom Portfolio") -> str:
    """
    빠른 백테스트 실행 함수
    
    Args:
        tickers: 종목 리스트
        weights: 비중 리스트 (None이면 동일가중)
        strategies: 전략 리스트
        rebalance_periods: 리밸런싱 주기 리스트
        start_date: 시작일
        end_date: 종료일
        portfolio_name: 포트폴리오 이름
    
    Returns:
        생성된 HTML 리포트 경로
    """
    analyzer = PortfolioAnalyzer()
    
    return (analyzer
            .load_data(start_date, end_date)
            .set_portfolio(tickers, weights, portfolio_name)
            .run_strategy_comparison(rebalance_periods, strategies)
            .generate_comprehensive_report())
