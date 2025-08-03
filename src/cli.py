"""
Stock Forecast Lab CLI 진입점

Typer를 사용한 통합 CLI 인터페이스를 제공합니다.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import settings

# Typer 앱 인스턴스
app = typer.Typer(
    name="stocklab",
    help="📈 Stock Forecast Lab - 주식 예측 백테스팅 시스템",
    add_completion=False,
)

# Rich Console
console = Console()

# 서브 명령어 그룹
ingest_app = typer.Typer(name="ingest", help="데이터 수집 명령어")
features_app = typer.Typer(name="features", help="피처 엔지니어링 명령어") 
backtest_app = typer.Typer(name="backtest", help="백테스트 명령어")
report_app = typer.Typer(name="report", help="리포트 생성 명령어")

app.add_typer(ingest_app, name="ingest")
app.add_typer(backtest_app, name="backtest")
app.add_typer(report_app, name="report")


@app.command()
def status():
    """프로젝트 상태를 확인합니다."""
    console.print("📊 [bold blue]Stock Forecast Lab 상태[/bold blue]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("항목", style="cyan")
    table.add_column("상태", style="green")
    table.add_column("경로/값", style="yellow")
    
    # 프로젝트 정보
    table.add_row("프로젝트 루트", "✅", str(settings.project_root))
    
    # 디렉토리 상태
    data_status = "✅" if settings.data_dir.exists() else "❌"
    table.add_row("데이터 디렉토리", data_status, str(settings.data_dir))
    
    raw_status = "✅" if settings.raw_data_dir.exists() else "❌"
    table.add_row("원본 데이터", raw_status, str(settings.raw_data_dir))
    
    silver_status = "✅" if settings.silver_data_dir.exists() else "❌"
    table.add_row("가공 데이터", silver_status, str(settings.silver_data_dir))
    
    # 데이터 파일 수
    if settings.raw_data_dir.exists():
        raw_files = len(list(settings.raw_data_dir.rglob("*.csv")))
        table.add_row("원본 파일 수", str(raw_files), "")
    
    if settings.silver_data_dir.exists():
        silver_files = len(list(settings.silver_data_dir.rglob("*.parquet")))
        table.add_row("가공 파일 수", str(silver_files), "")
    
    console.print(table)


@app.command("make-features")
def make_features(
    input_dir: str = typer.Option(
        str(settings.raw_data_dir),
        "--input", "-i",
        help="입력 데이터 디렉토리"
    ),
    output_dir: str = typer.Option(
        str(settings.silver_data_dir),
        "--output", "-o", 
        help="출력 데이터 디렉토리"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="기존 파일 덮어쓰기"
    )
):
    """피처 엔지니어링을 실행합니다."""
    console.print(f"🔧 [bold]피처 생성 시작[/bold]")
    console.print(f"입력: {input_dir}")
    console.print(f"출력: {output_dir}")
    
    try:
        from .features.ta_factors import create_features
        result = create_features(Path(input_dir), Path(output_dir), force=force)
        console.print(f"✅ [bold green]피처 생성 완료[/bold green]: {result}")
    except ImportError:
        console.print("❌ [bold red]피처 모듈을 찾을 수 없습니다[/bold red]")
    except Exception as e:
        console.print(f"❌ [bold red]피처 생성 실패[/bold red]: {e}")


# === 데이터 수집 서브 명령어 ===

@ingest_app.command("yfinance")
def ingest_yfinance(
    tickers: List[str] = typer.Option(
        ...,
        "--tickers", "-t",
        help="종목 코드 목록 (예: AAPL MSFT)"
    ),
    start: str = typer.Option(
        settings.default_start_date,
        "--start", "-s",
        help="시작 날짜 (YYYY-MM-DD)"
    ),
    end: str = typer.Option(
        datetime.now().strftime("%Y-%m-%d"),
        "--end", "-e", 
        help="종료 날짜 (YYYY-MM-DD)"
    ),
    interval: str = typer.Option(
        "1d",
        "--interval",
        help="데이터 간격 (1d, 1h 등)"
    )
):
    """Yahoo Finance에서 주식 데이터를 수집합니다."""
    console.print(f"📥 [bold]Yahoo Finance 데이터 수집 시작[/bold]")
    console.print(f"종목: {', '.join(tickers)}")
    console.print(f"기간: {start} ~ {end}")
    
    try:
        from .ingest.yfinance_cli import collect_data
        result = collect_data(tickers, start, end, interval)
        console.print(f"✅ [bold green]데이터 수집 완료[/bold green]: {len(result)} 종목")
    except ImportError:
        console.print("❌ [bold red]yfinance 모듈을 찾을 수 없습니다[/bold red]")
    except Exception as e:
        console.print(f"❌ [bold red]데이터 수집 실패[/bold red]: {e}")


# === 백테스트 서브 명령어 ===

@backtest_app.command("equal-weight")
def backtest_equal_weight(
    start: str = typer.Option(
        "2021-01-01",
        "--from",
        help="백테스트 시작 날짜"
    ),
    end: str = typer.Option(
        datetime.now().strftime("%Y-%m-%d"),
        "--to",
        help="백테스트 종료 날짜"
    ),
    rebalance: str = typer.Option(
        settings.default_rebalance_freq,
        "--rebalance", "-r",
        help="리밸런싱 주기"
    )
):
    """동일가중 전략 백테스트를 실행합니다."""
    console.print(f"🧪 [bold]동일가중 전략 백테스트 시작[/bold]")
    console.print(f"기간: {start} ~ {end}")
    
    try:
        from .strategies.equal_weight import weights
        from .backtest.engine import run_backtest
        
        result = run_backtest("equal_weight", start, end, rebalance)
        console.print(f"✅ [bold green]백테스트 완료[/bold green]")
        console.print(f"CAGR: {result.get('cagr', 0):.2%}")
        console.print(f"Sharpe: {result.get('sharpe', 0):.2f}")
    except ImportError:
        console.print("❌ [bold red]백테스트 모듈을 찾을 수 없습니다[/bold red]")
    except Exception as e:
        console.print(f"❌ [bold red]백테스트 실패[/bold red]: {e}")


@backtest_app.command("vol-parity")
def backtest_vol_parity(
    start: str = typer.Option(
        "2021-01-01",
        "--from",
        help="백테스트 시작 날짜"
    ),
    end: str = typer.Option(
        datetime.now().strftime("%Y-%m-%d"),
        "--to", 
        help="백테스트 종료 날짜"
    ),
    rebalance: str = typer.Option(
        settings.default_rebalance_freq,
        "--rebalance", "-r",
        help="리밸런싱 주기"
    )
):
    """변동성 패리티 전략 백테스트를 실행합니다."""
    console.print(f"🧪 [bold]변동성 패리티 전략 백테스트 시작[/bold]")
    console.print(f"기간: {start} ~ {end}")
    
    try:
        from .strategies.vol_parity import weights
        from .backtest.engine import run_backtest
        
        result = run_backtest("vol_parity", start, end, rebalance)
        console.print(f"✅ [bold green]백테스트 완료[/bold green]")
        console.print(f"CAGR: {result.get('cagr', 0):.2%}")
        console.print(f"Sharpe: {result.get('sharpe', 0):.2f}")
    except ImportError:
        console.print("❌ [bold red]백테스트 모듈을 찾을 수 없습니다[/bold red]")
    except Exception as e:
        console.print(f"❌ [bold red]백테스트 실패[/bold red]: {e}")


# === 리포트 서브 명령어 ===

@report_app.command("generate")
def generate_report(
    bt_id: Optional[str] = typer.Option(
        None,
        "--bt-id",
        help="백테스트 ID"
    ),
    latest: bool = typer.Option(
        False,
        "--latest",
        help="최신 백테스트 결과 사용"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="출력 파일 경로"
    )
):
    """백테스트 결과 리포트를 생성합니다."""
    console.print(f"📊 [bold]리포트 생성 시작[/bold]")
    
    try:
        from .reports.plotly_dash import generate_report as gen_report
        result = gen_report(bt_id, latest, output)
        console.print(f"✅ [bold green]리포트 생성 완료[/bold green]: {result}")
    except ImportError:
        console.print("❌ [bold red]리포트 모듈을 찾을 수 없습니다[/bold red]")
    except Exception as e:
        console.print(f"❌ [bold red]리포트 생성 실패[/bold red]: {e}")


@report_app.command("dashboard")
def start_dashboard(
    host: str = typer.Option(
        settings.dash_host,
        "--host",
        help="호스트 주소"
    ),
    port: int = typer.Option(
        settings.dash_port,
        "--port",
        help="포트 번호"
    ),
    debug: bool = typer.Option(
        settings.dash_debug,
        "--debug/--no-debug",
        help="디버그 모드"
    )
):
    """대시보드를 시작합니다."""
    console.print(f"🚀 [bold]대시보드 시작[/bold]")
    console.print(f"주소: http://{host}:{port}")
    
    try:
        from .reports.plotly_dash import start_dashboard as start_dash
        start_dash(host, port, debug)
    except ImportError:
        console.print("❌ [bold red]대시보드 모듈을 찾을 수 없습니다[/bold red]")
    except Exception as e:
        console.print(f"❌ [bold red]대시보드 시작 실패[/bold red]: {e}")


@app.command(name="yaml-analysis")
def yaml_analysis(
    config_file: str = typer.Argument(..., help="YAML 설정 파일 경로"),
    enhanced: bool = typer.Option(True, "--enhanced/--basic", help="향상된 분석 모드 (배당+실제거래)"),
    install_deps: bool = typer.Option(False, "--install", help="필요한 의존성 자동 설치")
):
    """
    YAML 설정 파일로 포트폴리오 분석을 실행합니다.
    
    향상된 모드에서는 배당 수익, 실제 주가, 거래 비용을 모두 고려합니다.
    """
    from pathlib import Path
    
    config_path = Path(config_file)
    
    if not config_path.exists():
        console.print(f"❌ [bold red]설정 파일을 찾을 수 없습니다[/bold red]: {config_file}")
        raise typer.Exit(code=1)
        
    # PyYAML 의존성 확인 및 설치
    if install_deps:
        console.print("📦 의존성 설치 중...")
        import subprocess
        try:
            subprocess.run(["poetry", "install"], check=True, capture_output=True)
            console.print("✅ [bold green]의존성 설치 완료[/bold green]")
        except subprocess.CalledProcessError as e:
            console.print(f"❌ [bold red]의존성 설치 실패[/bold red]: {e}")
            raise typer.Exit(code=1)
    
    mode_text = "향상된" if enhanced else "기본"
    console.print(f"📄 [bold blue]{mode_text} YAML 분석 시작[/bold blue]: {config_file}")
    
    try:
        if enhanced:
            from .config.enhanced_yaml_config import run_enhanced_yaml_analysis
            results = run_enhanced_yaml_analysis(str(config_path))
            summary = results['summary']
            
            console.print("\n🎉 [bold green]향상된 분석 완료![/bold green]")
            console.print(f"✅ 성공한 포트폴리오: {summary['success_count']} 개")
            console.print(f"❌ 실패한 포트폴리오: {summary['failed_count']} 개")
            console.print(f"📁 생성된 리포트: {len(summary['report_paths'])} 개")
            console.print(f"🏷️ 프로젝트: {summary['project_name']}")
            
            # 생성된 파일 목록
            if summary['report_paths']:
                console.print("\n📂 [bold blue]생성된 파일들:[/bold blue]")
                for report_path in summary['report_paths'][:8]:  # 처음 8개만
                    console.print(f"   • {report_path}")
                
                if len(summary['report_paths']) > 8:
                    console.print(f"   ... 외 {len(summary['report_paths'])-8} 개")
            
            console.print("\n💡 [bold cyan]향상된 기능:[/bold cyan]")
            console.print("   • 배당 수익 포함 총 수익률")
            console.print("   • 실제 주가 기반 주식 수 계산")
            console.print("   • 잔여 현금 및 거래 비용 추적")
            console.print("   • YAML 파일명 기반 독립 폴더")
        
        else:
            from .config.yaml_config import run_yaml_analysis
            results = run_yaml_analysis(str(config_path))
            summary = results['summary']
            
            console.print("\n🎉 [bold green]기본 분석 완료![/bold green]")
            console.print(f"✅ 성공한 포트폴리오: {summary['total_portfolios']} 개")
            console.print(f"❌ 실패한 포트폴리오: {summary['failed_portfolios']} 개")
            console.print(f"📁 생성된 리포트: {len(summary['generated_reports'])} 개")
            
            if summary['comparison_available']:
                console.print("📊 포트폴리오 비교 분석 포함")
            
            # 생성된 파일 목록
            if summary['generated_reports']:
                console.print("\n📂 [bold blue]생성된 파일들:[/bold blue]")
                for report_path in summary['generated_reports'][:5]:  # 처음 5개만
                    console.print(f"   • {report_path}")
                
                if len(summary['generated_reports']) > 5:
                    console.print(f"   ... 외 {len(summary['generated_reports'])-5} 개")
        
        console.print("✅ [bold green]YAML 분석 완료[/bold green]")
        
    except ImportError:
        console.print("❌ [bold red]PyYAML이 설치되지 않았습니다[/bold red]. --install 옵션을 사용하거나 'poetry install'을 실행하세요.")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"❌ [bold red]YAML 분석 실패[/bold red]: {e}")
        raise typer.Exit(code=1)


@app.command(name="create-yaml")
def create_yaml_template(
    output_file: str = typer.Option("my_portfolio.yaml", "--output", "-o", help="출력 파일명"),
    portfolio_type: str = typer.Option("simple", "--type", "-t", help="템플릿 타입: simple, multi")
):
    """
    YAML 설정 파일 템플릿을 생성합니다.
    """
    from pathlib import Path
    
    output_path = Path(output_file)
    
    templates = {
        "simple": """# 간단한 포트폴리오 분석 설정

global_settings:
  output_dir: "reports"
  start_date: "2022-01-01"
  end_date: "2024-12-31"
  transaction_cost: 0.001

portfolios:
  my_portfolio:
    name: "My Portfolio"
    description: "내 포트폴리오"
    
    assets:
      QQQ: 0.40    # 나스닥 100 (40%)
      VOO: 0.30    # S&P 500 (30%)
      GLD: 0.30    # 금 ETF (30%)
    
    strategies: ["equal_weight", "vol_parity"]
    rebalance_periods: ["3M", "6M"]

batch_analysis:
  enabled: true
  comparison:
    enabled: false
  reports:
    generate_individual: true
    include_charts: true
""",
        
        "multi": """# 다중 포트폴리오 비교 분석

global_settings:
  output_dir: "reports"
  start_date: "2022-01-01"
  end_date: "2024-12-31"
  transaction_cost: 0.001

portfolios:
  conservative:
    name: "Conservative Portfolio"
    description: "보수적 포트폴리오"
    assets:
      VOO: 0.60
      TLT: 0.40
    strategies: ["vol_parity"]
    rebalance_periods: ["6M", "1Y"]

  aggressive:
    name: "Aggressive Portfolio"
    description: "공격적 포트폴리오"
    assets:
      QQQ: 0.50
      BITO: 0.30
      ARKK: 0.20
    strategies: ["equal_weight", "vol_parity"]
    rebalance_periods: ["1M", "3M"]

batch_analysis:
  enabled: true
  comparison:
    enabled: true
    base_strategy: "equal_weight"
    base_period: "3M"
  reports:
    generate_individual: true
    generate_comparison: true
    include_charts: true
"""
    }
    
    if portfolio_type not in templates:
        console.print(f"❌ [bold red]알 수 없는 템플릿 타입[/bold red]: {portfolio_type}")
        console.print(f"사용 가능한 타입: {list(templates.keys())}")
        raise typer.Exit(code=1)
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(templates[portfolio_type], encoding='utf-8')
        
        console.print(f"✅ [bold green]YAML 템플릿 생성 완료[/bold green]: {output_path}")
        console.print(f"📝 다음 명령어로 분석 실행:")
        console.print(f"   [bold cyan]stocklab yaml-analysis {output_path}[/bold cyan]")
        
    except Exception as e:
        console.print(f"❌ [bold red]템플릿 생성 실패[/bold red]: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()