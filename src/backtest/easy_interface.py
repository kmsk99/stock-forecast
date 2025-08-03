"""
사용자 친화적 백테스트 인터페이스
간단한 함수 호출로 전체 분석 실행
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

from .portfolio_analyzer import PortfolioAnalyzer
from ..reports.advanced_reporter import AdvancedReporter


def quick_portfolio_analysis(
    tickers: List[str],
    weights: Optional[List[float]] = None,
    portfolio_name: str = "My Portfolio",
    strategies: List[str] = ['equal_weight', 'vol_parity'],
    rebalance_periods: List[str] = ['1M', '3M', '6M', '1Y'],
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    output_dir: str = "reports"
) -> Dict[str, str]:
    """
    원클릭 포트폴리오 분석 함수
    
    Args:
        tickers: 종목 리스트 (예: ['QQQ', 'VOO', 'BITO', 'GLD'])
        weights: 비중 리스트 (None이면 동일가중)
        portfolio_name: 포트폴리오 이름
        strategies: 분석할 전략 리스트
        rebalance_periods: 리밸런싱 주기 리스트
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        output_dir: 결과 저장 디렉토리
        
    Returns:
        생성된 파일 경로들 {'html': 'path', 'md': 'path', 'chart': 'path'}
    """
    
    logger.info(f"🚀 포트폴리오 분석 시작: {portfolio_name}")
    logger.info(f"   종목: {tickers}")
    logger.info(f"   전략: {strategies}")
    logger.info(f"   리밸런싱: {rebalance_periods}")
    
    try:
        # 1. 분석기 초기화 및 실행
        analyzer = PortfolioAnalyzer()
        
        analyzer.load_data(start_date, end_date)
        analyzer.set_portfolio(tickers, weights, portfolio_name)
        analyzer.run_strategy_comparison(rebalance_periods, strategies)
        
        # 2. 리포트 생성
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 차트 생성 (plt.show() 없이)
        chart_path = analyzer._generate_comprehensive_charts(output_path, timestamp)
        
        # 고급 리포트 생성
        reporter = AdvancedReporter(analyzer.results, analyzer.portfolio_specs)
        html_path = reporter.generate_html_report(output_path, timestamp)
        md_path = reporter.generate_markdown_report(output_path, timestamp)
        
        logger.success("✅ 포트폴리오 분석 완료!")
        logger.info(f"   📊 차트: {chart_path}")
        logger.info(f"   🌐 HTML: {html_path}")
        logger.info(f"   📄 MD: {md_path}")
        
        return {
            'html': str(html_path),
            'markdown': str(md_path),
            'chart': str(chart_path),
            'results': analyzer.results,
            'specs': analyzer.portfolio_specs
        }
        
    except Exception as e:
        logger.error(f"❌ 분석 실패: {e}")
        raise


def preset_portfolios():
    """사전 정의된 포트폴리오 모음"""
    
    return {
        "etf_4core": {
            "name": "4 Core ETF Portfolio",
            "tickers": ["QQQ", "VOO", "BITO", "GLD"],
            "weights": [0.3, 0.3, 0.2, 0.2],
            "description": "기술주(QQQ) + 대형주(VOO) + 비트코인(BITO) + 금(GLD)"
        },
        "classic_6040": {
            "name": "Classic 60/40 Portfolio", 
            "tickers": ["VOO", "TLT"],
            "weights": [0.6, 0.4],
            "description": "전통적인 주식 60% + 채권 40% 포트폴리오"
        },
        "global_diversified": {
            "name": "Global Diversified Portfolio",
            "tickers": ["VTI", "VXUS", "VGIT", "VNQ", "GLD"],
            "weights": [0.4, 0.2, 0.2, 0.1, 0.1],
            "description": "미국주식 + 해외주식 + 채권 + 리츠 + 금"
        },
        "growth_aggressive": {
            "name": "Aggressive Growth Portfolio",
            "tickers": ["QQQ", "ARKK", "BITO", "TQQQ"],
            "weights": [0.4, 0.3, 0.2, 0.1],
            "description": "고성장 기술주 중심의 공격적 포트폴리오"
        },
        "defensive": {
            "name": "Defensive Portfolio",
            "tickers": ["VOO", "TLT", "GLD", "VNQ"],
            "weights": [0.4, 0.3, 0.2, 0.1],
            "description": "방어적 자산 중심의 안정적 포트폴리오"
        }
    }


def analyze_preset_portfolio(
    preset_name: str,
    strategies: List[str] = ['equal_weight', 'vol_parity'],
    rebalance_periods: List[str] = ['1M', '3M', '6M'],
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31"
) -> Dict[str, str]:
    """
    사전 정의 포트폴리오 분석
    
    Args:
        preset_name: 프리셋 이름 ('etf_4core', 'classic_6040' 등)
        strategies: 분석할 전략
        rebalance_periods: 리밸런싱 주기
        start_date: 시작일
        end_date: 종료일
        
    Returns:
        분석 결과 파일 경로들
    """
    
    presets = preset_portfolios()
    
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"사용 가능한 프리셋: {available}")
    
    preset = presets[preset_name]
    
    logger.info(f"📋 프리셋 포트폴리오 분석: {preset['name']}")
    logger.info(f"   설명: {preset['description']}")
    
    return quick_portfolio_analysis(
        tickers=preset['tickers'],
        weights=preset['weights'],
        portfolio_name=preset['name'],
        strategies=strategies,
        rebalance_periods=rebalance_periods,
        start_date=start_date,
        end_date=end_date
    )


def compare_multiple_portfolios(
    portfolios: Dict[str, Dict[str, Any]],
    strategies: List[str] = ['equal_weight'],
    rebalance_period: str = '3M',
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31"
) -> Dict[str, Any]:
    """
    여러 포트폴리오 비교 분석
    
    Args:
        portfolios: 포트폴리오 딕셔너리 
                   {'name': {'tickers': [...], 'weights': [...]}}
        strategies: 분석할 전략
        rebalance_period: 리밸런싱 주기
        start_date: 시작일
        end_date: 종료일
        
    Returns:
        비교 분석 결과
    """
    
    logger.info(f"🔄 포트폴리오 비교 분석: {list(portfolios.keys())}")
    
    comparison_results = {}
    
    for name, portfolio in portfolios.items():
        logger.info(f"   분석 중: {name}")
        
        result = quick_portfolio_analysis(
            tickers=portfolio['tickers'],
            weights=portfolio.get('weights'),
            portfolio_name=name,
            strategies=strategies,
            rebalance_periods=[rebalance_period],
            start_date=start_date,
            end_date=end_date,
            output_dir=f"reports/comparison_{name.lower().replace(' ', '_')}"
        )
        
        comparison_results[name] = result
    
    # 비교 요약 생성
    summary = _generate_comparison_summary(comparison_results, strategies[0], rebalance_period)
    
    logger.success("✅ 포트폴리오 비교 완료!")
    return {
        'individual_results': comparison_results,
        'summary': summary
    }


def _generate_comparison_summary(
    results: Dict[str, Dict], 
    strategy: str, 
    period: str
) -> Dict[str, Any]:
    """포트폴리오 비교 요약 생성"""
    
    summary = {
        'rankings': {},
        'metrics': {}
    }
    
    # 각 지표별 랭킹
    metrics_to_rank = ['cagr', 'sharpe_ratio', 'max_drawdown', 'volatility']
    
    for metric in metrics_to_rank:
        portfolio_metrics = []
        
        for name, result in results.items():
            if 'results' in result and strategy in result['results']:
                value = result['results'][strategy][period]['metrics'][metric]
                portfolio_metrics.append((name, value))
        
        # 정렬 (드로우다운은 작을수록, 나머지는 클수록 좋음)
        reverse = True if metric != 'max_drawdown' else False
        portfolio_metrics.sort(key=lambda x: x[1], reverse=reverse)
        
        summary['rankings'][metric] = portfolio_metrics
        summary['metrics'][metric] = {name: value for name, value in portfolio_metrics}
    
    return summary


# 편의 함수들
def quick_etf_analysis(
    rebalance_periods: List[str] = ['1M', '3M', '6M']
) -> Dict[str, str]:
    """빠른 ETF 분석 (QQQ, VOO, BITO, GLD)"""
    return quick_portfolio_analysis(
        tickers=['QQQ', 'VOO', 'BITO', 'GLD'],
        portfolio_name="4 Core ETF Portfolio",
        rebalance_periods=rebalance_periods
    )


def quick_stock_analysis(
    tickers: List[str],
    portfolio_name: str = "Stock Portfolio"
) -> Dict[str, str]:
    """빠른 개별주식 분석"""
    return quick_portfolio_analysis(
        tickers=tickers,
        portfolio_name=portfolio_name,
        strategies=['equal_weight', 'vol_parity'],
        rebalance_periods=['1M', '3M', '6M']
    )


if __name__ == "__main__":
    # 테스트 실행
    print("🧪 간편 인터페이스 테스트")
    
    # 프리셋 포트폴리오 목록
    presets = preset_portfolios()
    print("\n📋 사용 가능한 프리셋 포트폴리오:")
    for name, info in presets.items():
        print(f"  - {name}: {info['description']}")
    
    print("\n사용 예시:")
    print("analyze_preset_portfolio('etf_4core')")
    print("quick_etf_analysis()")
    print("quick_stock_analysis(['AAPL', 'MSFT', 'GOOGL'])")