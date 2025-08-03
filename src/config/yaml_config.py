"""
YAML 설정 파일 기반 포트폴리오 분석 시스템
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from ..backtest.portfolio_analyzer import PortfolioAnalyzer
from ..reports.advanced_reporter import AdvancedReporter


class YAMLConfigAnalyzer:
    """YAML 설정 파일 기반 포트폴리오 분석기"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: YAML 설정 파일 경로
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = {}
        
        logger.info(f"📄 YAML 설정 로드: {config_path}")
    
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
    
    def run_analysis(self) -> Dict[str, Any]:
        """설정에 따라 분석 실행"""
        
        global_settings = self.config.get('global_settings', {})
        portfolios = self.config.get('portfolios', {})
        batch_settings = self.config.get('batch_analysis', {})
        
        # 실행할 포트폴리오 결정
        target_portfolios = batch_settings.get('target_portfolios', [])
        if not target_portfolios:
            target_portfolios = list(portfolios.keys())
        
        logger.info(f"🎯 분석 대상: {target_portfolios}")
        
        # 개별 포트폴리오 분석
        for portfolio_name in target_portfolios:
            if portfolio_name not in portfolios:
                logger.warning(f"⚠️ 포트폴리오 '{portfolio_name}' 설정을 찾을 수 없습니다.")
                continue
            
            logger.info(f"📊 분석 시작: {portfolio_name}")
            result = self._analyze_single_portfolio(
                portfolio_name, 
                portfolios[portfolio_name], 
                global_settings
            )
            self.results[portfolio_name] = result
        
        # 비교 분석 (옵션)
        if batch_settings.get('comparison', {}).get('enabled', False):
            logger.info("🔄 비교 분석 시작")
            comparison_result = self._generate_comparison_analysis(batch_settings['comparison'])
            self.results['_comparison'] = comparison_result
        
        logger.success("✅ 모든 분석 완료")
        return self.results
    
    def _analyze_single_portfolio(self, 
                                name: str, 
                                portfolio_config: Dict[str, Any], 
                                global_settings: Dict[str, Any]) -> Dict[str, Any]:
        """개별 포트폴리오 분석"""
        
        # 설정 추출
        assets = portfolio_config['assets']
        tickers = list(assets.keys())
        weights = list(assets.values())
        
        strategies = portfolio_config.get('strategies', global_settings.get('default_strategies', ['equal_weight']))
        rebalance_periods = portfolio_config.get('rebalance_periods', global_settings.get('default_rebalance_periods', ['1M', '3M']))
        
        start_date = global_settings.get('start_date', '2022-01-01')
        end_date = global_settings.get('end_date', '2024-12-31')
        
        # 포트폴리오 분석기 초기화
        analyzer = PortfolioAnalyzer(global_settings.get('data_path', 'data/silver/features.parquet'))
        
        try:
            # 분석 실행
            analyzer.load_data(start_date, end_date)
            analyzer.set_portfolio(tickers, weights, portfolio_config['name'])
            analyzer.run_strategy_comparison(rebalance_periods, strategies)
            
            # 리포트 생성 (설정에 따라)
            reports_config = self.config.get('batch_analysis', {}).get('reports', {})
            report_paths = {}
            
            if reports_config.get('generate_individual', True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(global_settings.get('output_dir', 'reports')) / f"{name}_{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 차트 생성
                if reports_config.get('include_charts', True):
                    chart_path = analyzer._generate_comprehensive_charts(output_dir, timestamp)
                    report_paths['chart'] = str(chart_path)
                
                # HTML/MD 리포트 생성
                reporter = AdvancedReporter(analyzer.results, analyzer.portfolio_specs)
                html_path = reporter.generate_html_report(output_dir, timestamp)
                md_path = reporter.generate_markdown_report(output_dir, timestamp)
                
                report_paths['html'] = str(html_path)
                report_paths['markdown'] = str(md_path)
            
            return {
                'config': portfolio_config,
                'analyzer_results': analyzer.results,
                'portfolio_specs': analyzer.portfolio_specs,
                'report_paths': report_paths,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"❌ {name} 분석 실패: {e}")
            return {
                'config': portfolio_config,
                'error': str(e),
                'status': 'failed'
            }
    
    def _generate_comparison_analysis(self, comparison_config: Dict[str, Any]) -> Dict[str, Any]:
        """비교 분석 생성"""
        
        base_strategy = comparison_config.get('base_strategy', 'equal_weight')
        base_period = comparison_config.get('base_period', '3M')
        
        comparison_data = {}
        
        for name, result in self.results.items():
            if result['status'] != 'success':
                continue
                
            analyzer_results = result['analyzer_results']
            
            if base_strategy in analyzer_results and base_period in analyzer_results[base_strategy]:
                metrics = analyzer_results[base_strategy][base_period]['metrics']
                comparison_data[name] = {
                    'cagr': metrics['cagr'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'volatility': metrics['volatility'],
                    'win_rate': metrics['win_rate']
                }
        
        # 랭킹 생성
        rankings = {}
        for metric in ['cagr', 'sharpe_ratio', 'max_drawdown', 'volatility', 'win_rate']:
            reverse = False if metric == 'max_drawdown' else True
            sorted_portfolios = sorted(comparison_data.items(), 
                                     key=lambda x: x[1][metric], 
                                     reverse=reverse)
            rankings[metric] = [(name, value[metric]) for name, value in sorted_portfolios]
        
        # 비교 리포트 생성
        comparison_report = self._create_comparison_report(comparison_data, rankings)
        
        return {
            'data': comparison_data,
            'rankings': rankings,
            'report': comparison_report,
            'base_strategy': base_strategy,
            'base_period': base_period
        }
    
    def _create_comparison_report(self, data: Dict[str, Dict], rankings: Dict[str, List]) -> str:
        """비교 리포트 마크다운 생성"""
        
        report = f"""# 📊 포트폴리오 비교 분석 리포트

## 🏆 종합 랭킹

### 📈 수익률 (CAGR) 랭킹
"""
        for i, (name, value) in enumerate(rankings['cagr'], 1):
            report += f"{i}. **{name}**: {value:.2%}\n"
        
        report += f"""
### ⚖️ 샤프 비율 랭킹
"""
        for i, (name, value) in enumerate(rankings['sharpe_ratio'], 1):
            report += f"{i}. **{name}**: {value:.3f}\n"
        
        report += f"""
### 🛡️ 안정성 (최대 낙폭) 랭킹
"""
        for i, (name, value) in enumerate(rankings['max_drawdown'], 1):
            report += f"{i}. **{name}**: {value:.2%}\n"
        
        report += f"""
## 📊 상세 비교표

| 포트폴리오 | CAGR | 샤프비율 | 최대낙폭 | 변동성 | 승률 |
|-----------|------|----------|----------|--------|------|
"""
        
        for name, metrics in data.items():
            report += f"| {name} | {metrics['cagr']:.2%} | {metrics['sharpe_ratio']:.3f} | {metrics['max_drawdown']:.2%} | {metrics['volatility']:.2%} | {metrics['win_rate']:.2%} |\n"
        
        report += f"""
## 💡 결론

**최고 수익률**: {rankings['cagr'][0][0]} ({rankings['cagr'][0][1]:.2%})
**최고 샤프비율**: {rankings['sharpe_ratio'][0][0]} ({rankings['sharpe_ratio'][0][1]:.3f})
**최고 안정성**: {rankings['max_drawdown'][0][0]} ({rankings['max_drawdown'][0][1]:.2%})

---
*Generated by Stock Forecast Lab on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def save_comparison_report(self, output_path: Optional[str] = None) -> str:
        """비교 리포트 저장"""
        
        if '_comparison' not in self.results:
            logger.warning("⚠️ 비교 분석 결과가 없습니다.")
            return ""
        
        if output_path is None:
            output_path = Path(self.config['global_settings'].get('output_dir', 'reports'))
            output_path = output_path / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.results['_comparison']['report'])
        
        logger.success(f"📄 비교 리포트 저장: {output_path}")
        return str(output_path)
    
    def get_summary(self) -> Dict[str, Any]:
        """분석 결과 요약"""
        
        summary = {
            'total_portfolios': len([r for r in self.results.values() if r.get('status') == 'success']),
            'failed_portfolios': len([r for r in self.results.values() if r.get('status') == 'failed']),
            'generated_reports': [],
            'comparison_available': '_comparison' in self.results
        }
        
        for name, result in self.results.items():
            if name.startswith('_'):
                continue
                
            if result['status'] == 'success' and 'report_paths' in result:
                summary['generated_reports'].extend(result['report_paths'].values())
        
        return summary


def run_yaml_analysis(config_path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일로 포트폴리오 분석 실행
    
    Args:
        config_path: YAML 설정 파일 경로
        
    Returns:
        분석 결과 딕셔너리
    """
    
    analyzer = YAMLConfigAnalyzer(config_path)
    results = analyzer.run_analysis()
    
    # 비교 리포트 저장 (설정된 경우)
    if analyzer.config.get('batch_analysis', {}).get('comparison', {}).get('enabled', False):
        analyzer.save_comparison_report()
    
    # 요약 출력
    summary = analyzer.get_summary()
    logger.info(f"📊 분석 완료 요약:")
    logger.info(f"   ✅ 성공: {summary['total_portfolios']} 포트폴리오")
    logger.info(f"   ❌ 실패: {summary['failed_portfolios']} 포트폴리오") 
    logger.info(f"   📁 생성된 리포트: {len(summary['generated_reports'])} 개")
    
    return {
        'results': results,
        'summary': summary,
        'analyzer': analyzer
    }


if __name__ == "__main__":
    # 테스트 실행
    print("🧪 YAML 설정 분석기 테스트")
    
    config_path = "configs/portfolio_config_example.yaml"
    if Path(config_path).exists():
        print(f"📄 설정 파일 발견: {config_path}")
        
        try:
            results = run_yaml_analysis(config_path)
            print("✅ 테스트 성공!")
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
    else:
        print(f"⚠️ 설정 파일이 없습니다: {config_path}")
        print("먼저 예시 설정 파일을 생성하세요.")