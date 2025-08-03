"""
향상된 YAML 설정 파일 기반 포트폴리오 분석 시스템
- YAML별 독립 폴더 구조
- 배당 수익 포함 총 수익률
- 실제 주가/예산/주식 수 고려
"""

import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from ..backtest.portfolio_analyzer import PortfolioAnalyzer
from ..reports.advanced_reporter import AdvancedReporter


class EnhancedYAMLAnalyzer:
    """향상된 YAML 설정 파일 기반 포트폴리오 분석기"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: YAML 설정 파일 경로
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = {}
        
        # YAML 파일명에서 프로젝트 이름 추출
        self.project_name = self.config_path.stem
        
        logger.info(f"📄 Enhanced YAML 설정 로드: {config_path}")
        logger.info(f"🏷️ 프로젝트 이름: {self.project_name}")
    
    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.success("✅ YAML 설정 로드 완료")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML 파싱 오류: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 설정 파일 로드 실패: {e}")
            raise
    
    def _get_dividend_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """배당 데이터 수집"""
        
        logger.info(f"💰 배당 데이터 수집: {len(tickers)} 종목")
        
        dividend_data = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # 배당 데이터 가져오기
                dividends = stock.dividends
                
                if not dividends.empty:
                    # 날짜 범위 필터링
                    dividends = dividends[
                        (dividends.index >= start_date) & 
                        (dividends.index <= end_date)
                    ]
                    
                    dividend_data[ticker] = dividends
                    logger.debug(f"   📊 {ticker}: {len(dividends)} 배당 지급")
                else:
                    logger.debug(f"   📊 {ticker}: 배당 없음")
                    dividend_data[ticker] = pd.Series(dtype=float)
                    
            except Exception as e:
                logger.warning(f"⚠️ {ticker} 배당 데이터 수집 실패: {e}")
                dividend_data[ticker] = pd.Series(dtype=float)
        
        return dividend_data
    
    def _calculate_total_returns(self, 
                               prices: pd.DataFrame, 
                               dividend_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """배당을 포함한 총 수익률 계산"""
        
        logger.info("📈 총 수익률 계산 (배당 포함)")
        
        # 일일 수익률
        price_returns = prices.pct_change().fillna(0)
        
        # 배당 수익률 계산
        dividend_returns = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        for ticker in prices.columns:
            if ticker in dividend_data and not dividend_data[ticker].empty:
                dividends = dividend_data[ticker]
                
                for div_date, div_amount in dividends.items():
                    # div_date를 Timestamp로 변환
                    div_date = pd.Timestamp(div_date)
                    
                    # 배당 지급일에 가장 가까운 거래일 찾기
                    closest_date = prices.index[prices.index <= div_date]
                    
                    if len(closest_date) > 0:
                        closest_date = closest_date[-1]
                        
                        # 전일 종가 기준 배당 수익률
                        prev_idx = prices.index.get_loc(closest_date)
                        if prev_idx > 0:
                            prev_price = prices[ticker].iloc[prev_idx - 1]
                            if prev_price > 0:
                                dividend_yield = div_amount / prev_price
                                dividend_returns.loc[closest_date, ticker] = dividend_yield
        
        # 총 수익률 = 가격 수익률 + 배당 수익률
        total_returns = price_returns + dividend_returns
        
        dividend_summary = {}
        for ticker in prices.columns:
            total_div_yield = dividend_returns[ticker].sum()
            dividend_summary[ticker] = total_div_yield * 100  # 퍼센트로 변환
        
        logger.info(f"💰 연간 배당 수익률 요약:")
        for ticker, yield_pct in dividend_summary.items():
            logger.info(f"   {ticker}: {yield_pct:.2f}%")
        
        return total_returns, dividend_summary
    
    def _simulate_realistic_portfolio(self, 
                                    prices: pd.DataFrame,
                                    weights: np.ndarray,
                                    initial_capital: float,
                                    rebalance_dates: List[pd.Timestamp],
                                    transaction_cost: float,
                                    total_returns: pd.DataFrame) -> Dict[str, Any]:
        """실제 주가와 예산을 고려한 포트폴리오 시뮬레이션"""
        
        logger.info(f"💼 현실적 포트폴리오 시뮬레이션 시작")
        logger.info(f"   초기 자본: ${initial_capital:,.0f}")
        
        # 결과 저장용
        portfolio_history = []
        holdings_history = []
        cash_history = []
        transaction_history = []
        
        current_cash = initial_capital
        current_holdings = pd.Series(0, index=prices.columns)  # 보유 주식 수
        
        for i, date in enumerate(prices.index):
            current_prices = prices.loc[date]
            daily_returns = total_returns.loc[date] if date in total_returns.index else pd.Series(0, index=prices.columns)
            
            # 리밸런싱 체크
            is_rebalance_day = date in rebalance_dates or i == 0
            
            if is_rebalance_day:
                logger.debug(f"   📅 리밸런싱: {date.date()}")
                
                # 현재 포트폴리오 가치 계산
                current_holdings_value = (current_holdings * current_prices).sum()
                total_portfolio_value = current_cash + current_holdings_value
                
                # 목표 배분 계산
                target_values = total_portfolio_value * weights
                target_shares = (target_values / current_prices).round().astype(int)  # 정수 주식 수
                
                # 거래 실행
                transactions = {}
                total_transaction_cost = 0
                
                for ticker in prices.columns:
                    current_shares = current_holdings[ticker]
                    target_shares_ticker = target_shares[ticker]
                    shares_diff = target_shares_ticker - current_shares
                    
                    if shares_diff != 0:
                        trade_value = abs(shares_diff) * current_prices[ticker]
                        cost = trade_value * transaction_cost
                        
                        if shares_diff > 0:  # 매수
                            total_cost = trade_value + cost
                            if current_cash >= total_cost:
                                current_cash -= total_cost
                                current_holdings[ticker] = target_shares_ticker
                                total_transaction_cost += cost
                                transactions[ticker] = {'action': 'BUY', 'shares': shares_diff, 'cost': cost}
                            else:
                                # 현금 부족 시 가능한 만큼만 매수
                                affordable_shares = int((current_cash - cost) / current_prices[ticker])
                                if affordable_shares > 0:
                                    actual_cost = affordable_shares * current_prices[ticker] + cost
                                    current_cash -= actual_cost
                                    current_holdings[ticker] = current_shares + affordable_shares
                                    total_transaction_cost += cost
                                    transactions[ticker] = {'action': 'BUY', 'shares': affordable_shares, 'cost': cost}
                        
                        else:  # 매도
                            sell_proceeds = trade_value - cost
                            current_cash += sell_proceeds
                            current_holdings[ticker] = target_shares_ticker
                            total_transaction_cost += cost
                            transactions[ticker] = {'action': 'SELL', 'shares': abs(shares_diff), 'cost': cost}
                
                transaction_history.append({
                    'date': date,
                    'transactions': transactions,
                    'total_cost': total_transaction_cost
                })
            
            # 배당 수익 반영 (현금으로 받음)
            if date in total_returns.index:
                dividend_income = 0
                for ticker in prices.columns:
                    if daily_returns[ticker] > 0 and current_holdings[ticker] > 0:
                        # 배당 부분만 추출 (총 수익률에서 가격 수익률 제외)
                        price_return = prices.loc[date, ticker] / prices.shift(1).loc[date, ticker] - 1 if i > 0 else 0
                        dividend_return = daily_returns[ticker] - price_return
                        
                        if dividend_return > 0:
                            dividend_amount = current_holdings[ticker] * prices.loc[date, ticker] * dividend_return
                            dividend_income += dividend_amount
                
                current_cash += dividend_income
            
            # 포트폴리오 가치 계산
            holdings_value = (current_holdings * current_prices).sum()
            total_value = current_cash + holdings_value
            
            # 기록 저장
            portfolio_history.append({
                'date': date,
                'total_value': total_value,
                'cash': current_cash,
                'holdings_value': holdings_value,
                'cash_ratio': current_cash / total_value if total_value > 0 else 0
            })
            
            holdings_history.append({
                'date': date,
                **{f'{ticker}_shares': current_holdings[ticker] for ticker in prices.columns},
                **{f'{ticker}_value': current_holdings[ticker] * current_prices[ticker] for ticker in prices.columns}
            })
            
            cash_history.append({
                'date': date,
                'cash': current_cash
            })
        
        # 결과 정리
        portfolio_df = pd.DataFrame(portfolio_history).set_index('date')
        holdings_df = pd.DataFrame(holdings_history).set_index('date')
        
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # 총 거래 비용
        total_transaction_costs = sum([t['total_cost'] for t in transaction_history])
        
        result = {
            'portfolio_values': portfolio_df,
            'holdings_history': holdings_df,
            'transaction_history': transaction_history,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_transaction_costs': total_transaction_costs,
            'final_cash': current_cash,
            'final_holdings': current_holdings.to_dict()
        }
        
        logger.info(f"💼 시뮬레이션 완료:")
        logger.info(f"   최종 가치: ${final_value:,.0f}")
        logger.info(f"   총 수익률: {total_return:.2%}")
        logger.info(f"   거래 비용: ${total_transaction_costs:.2f}")
        logger.info(f"   잔여 현금: ${current_cash:,.2f}")
        
        return result
    
    def _analyze_single_portfolio_enhanced(self, 
                                         name: str, 
                                         portfolio_config: Dict[str, Any], 
                                         global_settings: Dict[str, Any]) -> Dict[str, Any]:
        """향상된 개별 포트폴리오 분석"""
        
        # 설정 추출
        assets = portfolio_config['assets']
        tickers = list(assets.keys())
        weights = np.array(list(assets.values()))
        
        strategies = portfolio_config.get('strategies', global_settings.get('default_strategies', ['equal_weight']))
        rebalance_periods = portfolio_config.get('rebalance_periods', global_settings.get('default_rebalance_periods', ['1M', '3M']))
        
        start_date = global_settings.get('start_date', '2022-01-01')
        end_date = global_settings.get('end_date', '2024-12-31')
        transaction_cost = global_settings.get('transaction_cost', 0.001)
        initial_capital = portfolio_config.get('initial_capital', global_settings.get('initial_capital', 100000))
        
        logger.info(f"💼 {name} 향상된 분석 시작")
        
        try:
            # 1. 기본 데이터 로드 (기존 피처 데이터)
            data_path = global_settings.get('data_path', 'data/silver/features.parquet')
            df = pd.read_parquet(data_path).reset_index()
            
            # 날짜 필터링 및 종목 필터링
            df_filtered = df[
                (df['date'] >= start_date) & 
                (df['date'] <= end_date) &
                (df['ticker'].isin(tickers))
            ].copy()
            
            # 가격 데이터 피벗
            prices = df_filtered.pivot(index='date', columns='ticker', values='close')
            prices = prices[tickers].dropna()  # 순서 보장 및 결측치 제거
            
            # 2. 배당 데이터 수집
            dividend_data = self._get_dividend_data(tickers, start_date, end_date)
            
            # 3. 총 수익률 계산 (배당 포함)
            total_returns, dividend_summary = self._calculate_total_returns(prices, dividend_data)
            
            # 4. 전략별 분석
            enhanced_results = {}
            
            for strategy_name in strategies:
                enhanced_results[strategy_name] = {}
                
                for rebalance_period in rebalance_periods:
                    logger.info(f"   📊 {strategy_name} - {rebalance_period}")
                    
                    # 리밸런싱 날짜 생성
                    rebalance_dates = self._get_rebalance_dates(prices.index, rebalance_period)
                    
                    # 전략에 따른 가중치 결정
                    if strategy_name == 'equal_weight':
                        strategy_weights = np.array([1.0 / len(tickers)] * len(tickers))
                    elif strategy_name == 'vol_parity':
                        # 간단한 변동성 패리티 (20일 변동성 기준)
                        vol = total_returns.rolling(20).std().dropna()
                        if not vol.empty:
                            inverse_vol = 1 / vol.iloc[-20:].mean()  # 최근 20일 평균 역변동성
                            strategy_weights = inverse_vol / inverse_vol.sum()
                            strategy_weights = strategy_weights.values
                        else:
                            strategy_weights = weights  # 폴백
                    else:
                        strategy_weights = weights  # 커스텀 가중치
                    
                    # 현실적 포트폴리오 시뮬레이션
                    simulation_result = self._simulate_realistic_portfolio(
                        prices, strategy_weights, initial_capital, 
                        rebalance_dates, transaction_cost, total_returns
                    )
                    
                    # 기본 성과 지표 계산
                    portfolio_returns = simulation_result['portfolio_values']['total_value'].pct_change().fillna(0)
                    
                    enhanced_results[strategy_name][rebalance_period] = {
                        'simulation_result': simulation_result,
                        'dividend_summary': dividend_summary,
                        'strategy_weights': strategy_weights,
                        'rebalance_dates': rebalance_dates,
                        'portfolio_returns': portfolio_returns,
                        'metrics': self._calculate_enhanced_metrics(simulation_result, portfolio_returns)
                    }
            
            # 5. 프로젝트별 폴더에 리포트 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = Path(global_settings.get('output_dir', 'reports'))
            project_folder = output_base / self.project_name  # YAML 파일명 기반 폴더
            output_dir = project_folder / f"{name}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 리포트 생성
            report_paths = self._generate_enhanced_reports(
                enhanced_results, portfolio_config, output_dir, timestamp, dividend_summary
            )
            
            return {
                'config': portfolio_config,
                'enhanced_results': enhanced_results,
                'dividend_summary': dividend_summary,
                'report_paths': report_paths,
                'output_dir': str(output_dir),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"❌ {name} 향상된 분석 실패: {e}")
            return {
                'config': portfolio_config,
                'error': str(e),
                'status': 'failed'
            }
    
    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex, period: str) -> List[pd.Timestamp]:
        """리밸런싱 날짜 생성"""
        dates = [date_index[0]]  # 첫째 날
        
        if period == '1D':
            return list(date_index)
        
        freq_map = {'1W': 'W', '1M': 'MS', '3M': '3MS', '6M': '6MS', '1Y': 'YS'}
        freq = freq_map.get(period, 'MS')
        
        period_dates = pd.date_range(start=date_index[0], end=date_index[-1], freq=freq)
        
        for date in period_dates:
            future_dates = date_index[date_index >= date]
            if len(future_dates) > 0:
                actual_date = future_dates[0]
                if actual_date not in dates:
                    dates.append(actual_date)
        
        return sorted(dates)
    
    def _calculate_enhanced_metrics(self, simulation_result: Dict, portfolio_returns: pd.Series) -> Dict:
        """향상된 성과 지표 계산"""
        
        initial_capital = simulation_result['initial_capital']
        final_value = simulation_result['final_value']
        total_return = simulation_result['total_return']
        
        n_years = len(portfolio_returns) / 252
        cagr = (final_value / initial_capital) ** (1/n_years) - 1 if n_years > 0 else 0
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # 샤프 비율
        risk_free_rate = 0.02
        excess_returns = portfolio_returns - risk_free_rate/252
        sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        
        # 드로우다운
        cumulative = simulation_result['portfolio_values']['total_value'] / initial_capital
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 실제 거래 정보
        transaction_costs = simulation_result['total_transaction_costs']
        final_cash = simulation_result['final_cash']
        cash_ratio = final_cash / final_value if final_value > 0 else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'transaction_costs': transaction_costs,
            'final_cash': final_cash,
            'cash_ratio': cash_ratio,
            'cost_ratio': transaction_costs / initial_capital
        }
    
    def _generate_enhanced_reports(self, 
                                 enhanced_results: Dict,
                                 portfolio_config: Dict,
                                 output_dir: Path,
                                 timestamp: str,
                                 dividend_summary: Dict) -> Dict[str, str]:
        """향상된 리포트 생성"""
        
        logger.info(f"📄 향상된 리포트 생성: {output_dir}")
        
        # 1. 상세 CSV 리포트
        csv_path = output_dir / f"detailed_results_{timestamp}.csv"
        self._save_detailed_csv(enhanced_results, csv_path)
        
        # 2. 배당 요약 리포트
        dividend_path = output_dir / f"dividend_summary_{timestamp}.csv"
        pd.DataFrame([dividend_summary]).to_csv(dividend_path, index=False)
        
        # 3. 거래 내역 리포트
        trade_path = output_dir / f"trading_history_{timestamp}.csv"
        self._save_trading_history(enhanced_results, trade_path)
        
        # 4. 마크다운 종합 리포트
        md_path = output_dir / f"comprehensive_report_{timestamp}.md"
        self._generate_enhanced_markdown(enhanced_results, portfolio_config, dividend_summary, md_path)
        
        return {
            'detailed_csv': str(csv_path),
            'dividend_csv': str(dividend_path),
            'trading_csv': str(trade_path),
            'markdown': str(md_path)
        }
    
    def _save_detailed_csv(self, enhanced_results: Dict, csv_path: Path):
        """상세 결과 CSV 저장"""
        
        detailed_data = []
        
        for strategy in enhanced_results:
            for period in enhanced_results[strategy]:
                result = enhanced_results[strategy][period]
                metrics = result['metrics']
                
                row = {
                    'Strategy': strategy,
                    'Rebalance_Period': period,
                    'Final_Value': metrics['final_value'],
                    'Total_Return': metrics['total_return'],
                    'CAGR': metrics['cagr'],
                    'Volatility': metrics['volatility'],
                    'Sharpe_Ratio': metrics['sharpe_ratio'],
                    'Max_Drawdown': metrics['max_drawdown'],
                    'Transaction_Costs': metrics['transaction_costs'],
                    'Final_Cash': metrics['final_cash'],
                    'Cash_Ratio': metrics['cash_ratio'],
                    'Cost_Ratio': metrics['cost_ratio']
                }
                detailed_data.append(row)
        
        pd.DataFrame(detailed_data).to_csv(csv_path, index=False)
        logger.info(f"💾 상세 결과 저장: {csv_path}")
    
    def _save_trading_history(self, enhanced_results: Dict, trade_path: Path):
        """거래 내역 CSV 저장"""
        
        # 첫 번째 전략/기간의 거래 내역만 저장 (예시)
        for strategy in enhanced_results:
            for period in enhanced_results[strategy]:
                transactions = enhanced_results[strategy][period]['simulation_result']['transaction_history']
                
                trade_data = []
                for trans in transactions:
                    if trans['transactions']:  # 거래가 있는 경우만
                        for ticker, details in trans['transactions'].items():
                            trade_data.append({
                                'Date': trans['date'].date(),
                                'Ticker': ticker,
                                'Action': details['action'],
                                'Shares': details['shares'],
                                'Cost': details['cost']
                            })
                
                if trade_data:
                    pd.DataFrame(trade_data).to_csv(trade_path, index=False)
                    logger.info(f"📋 거래 내역 저장: {trade_path}")
                break
            break
    
    def _generate_enhanced_markdown(self, 
                                  enhanced_results: Dict,
                                  portfolio_config: Dict,
                                  dividend_summary: Dict,
                                  md_path: Path):
        """향상된 마크다운 리포트 생성"""
        
        report = f"""# 📊 향상된 포트폴리오 분석 리포트

## 📋 포트폴리오 구성

**이름**: {portfolio_config['name']}
**설명**: {portfolio_config['description']}

### 자산 배분
"""
        
        for ticker, weight in portfolio_config['assets'].items():
            report += f"- **{ticker}**: {weight:.1%}\n"
        
        report += f"""

### 💰 배당 수익률 (연간)
"""
        
        for ticker, div_yield in dividend_summary.items():
            report += f"- **{ticker}**: {div_yield:.2f}%\n"
        
        report += f"""

## 📈 전략별 성과 분석

"""
        
        for strategy in enhanced_results:
            report += f"### {strategy.replace('_', ' ').title()} 전략\n\n"
            
            # 테이블 헤더
            report += "| 리밸런싱 | 최종가치 | 총수익률 | CAGR | 샤프비율 | 최대낙폭 | 거래비용 | 잔여현금 |\n"
            report += "|---------|----------|----------|------|----------|----------|----------|----------|\n"
            
            for period in enhanced_results[strategy]:
                metrics = enhanced_results[strategy][period]['metrics']
                report += f"| {period} | ${metrics['final_value']:,.0f} | {metrics['total_return']:.2%} | {metrics['cagr']:.2%} | {metrics['sharpe_ratio']:.3f} | {metrics['max_drawdown']:.2%} | ${metrics['transaction_costs']:.0f} | ${metrics['final_cash']:,.0f} |\n"
            
            report += "\n"
        
        report += f"""

## 💡 핵심 인사이트

### 배당 효과
- 배당을 포함한 총 수익률로 분석하여 보다 현실적인 결과 제공
- 배당 재투자 효과가 장기 수익률에 미치는 영향 고려

### 실제 거래 반영
- 정수 주식 수만 거래 (fractional shares 미지원 가정)
- 거래 비용이 수익률에 미치는 실제 영향 분석
- 잔여 현금 관리 및 현금 비율 추적

### 리밸런싱 효과
- 리밸런싱 주기에 따른 거래 비용과 수익률의 트레이드오프
- 실제 거래 가능한 날짜에만 리밸런싱 실행

---
*Generated by Enhanced Stock Forecast Lab on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📄 종합 리포트 저장: {md_path}")
    
    def run_enhanced_analysis(self) -> Dict[str, Any]:
        """향상된 분석 실행"""
        
        global_settings = self.config.get('global_settings', {})
        portfolios = self.config.get('portfolios', {})
        batch_settings = self.config.get('batch_analysis', {})
        
        # 실행할 포트폴리오 결정
        target_portfolios = batch_settings.get('target_portfolios', [])
        if not target_portfolios:
            target_portfolios = list(portfolios.keys())
        
        logger.info(f"🎯 향상된 분석 대상: {target_portfolios}")
        
        # 개별 포트폴리오 분석
        for portfolio_name in target_portfolios:
            if portfolio_name not in portfolios:
                logger.warning(f"⚠️ 포트폴리오 '{portfolio_name}' 설정을 찾을 수 없습니다.")
                continue
            
            result = self._analyze_single_portfolio_enhanced(
                portfolio_name, 
                portfolios[portfolio_name], 
                global_settings
            )
            self.results[portfolio_name] = result
        
        logger.success("✅ 향상된 분석 완료")
        return self.results


def run_enhanced_yaml_analysis(config_path: str) -> Dict[str, Any]:
    """
    향상된 YAML 설정 파일로 포트폴리오 분석 실행
    
    Args:
        config_path: YAML 설정 파일 경로
        
    Returns:
        향상된 분석 결과 딕셔너리
    """
    
    analyzer = EnhancedYAMLAnalyzer(config_path)
    results = analyzer.run_enhanced_analysis()
    
    # 요약 출력
    success_count = len([r for r in results.values() if r.get('status') == 'success'])
    failed_count = len([r for r in results.values() if r.get('status') == 'failed'])
    
    logger.info(f"🎉 향상된 분석 완료 요약:")
    logger.info(f"   ✅ 성공: {success_count} 포트폴리오")
    logger.info(f"   ❌ 실패: {failed_count} 포트폴리오")
    
    # 생성된 리포트 경로들
    all_report_paths = []
    for result in results.values():
        if result.get('status') == 'success' and 'report_paths' in result:
            all_report_paths.extend(result['report_paths'].values())
    
    logger.info(f"   📁 생성된 리포트: {len(all_report_paths)} 개")
    
    return {
        'results': results,
        'analyzer': analyzer,
        'summary': {
            'success_count': success_count,
            'failed_count': failed_count,
            'report_paths': all_report_paths,
            'project_name': analyzer.project_name
        }
    }


if __name__ == "__main__":
    # 테스트 실행
    print("🧪 향상된 YAML 설정 분석기 테스트")
    
    config_path = "test_portfolio.yaml"
    if Path(config_path).exists():
        print(f"📄 설정 파일 발견: {config_path}")
        
        try:
            results = run_enhanced_yaml_analysis(config_path)
            print("✅ 테스트 성공!")
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
    else:
        print(f"⚠️ 설정 파일이 없습니다: {config_path}")
        print("먼저 설정 파일을 생성하세요.")