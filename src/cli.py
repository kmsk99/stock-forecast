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


if __name__ == "__main__":
    app()