"""
Plotly/Dash 기반 대시보드 및 리포트 생성 모듈

백테스트 결과를 시각화하고 인터랙티브 대시보드를 제공합니다.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings

import pandas as pd
import numpy as np
from loguru import logger

# Plotly imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
    logger.info("Plotly 사용 가능")
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly가 설치되지 않았습니다. 기본 차트만 사용됩니다.")

# Dash imports
try:
    import dash
    from dash import dcc, html, Input, Output, callback
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
    logger.info("Dash 사용 가능")
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Dash가 설치되지 않았습니다. 대시보드 기능을 사용할 수 없습니다.")

from ..config import settings
from ..utils.paths import get_report_path, get_backtest_result_path
from ..backtest.engine import load_backtest_result, list_backtest_results


def create_performance_chart(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "포트폴리오 성과"
) -> go.Figure:
    """성과 차트를 생성합니다.
    
    Args:
        returns: 포트폴리오 수익률
        benchmark_returns: 벤치마크 수익률
        title: 차트 제목
        
    Returns:
        Plotly Figure 객체
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly가 설치되지 않았습니다.")
    
    # 누적 수익률 계산
    cumulative_returns = (1 + returns).cumprod()
    
    fig = go.Figure()
    
    # 포트폴리오 수익률
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns.values,
        mode='lines',
        name='포트폴리오',
        line=dict(width=2, color='#1f77b4')
    ))
    
    # 벤치마크 수익률
    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            mode='lines',
            name='벤치마크',
            line=dict(width=2, color='#ff7f0e', dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='날짜',
        yaxis_title='누적 수익률',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0, y=1)
    )
    
    return fig


def create_drawdown_chart(returns: pd.Series, title: str = "드로우다운") -> go.Figure:
    """드로우다운 차트를 생성합니다.
    
    Args:
        returns: 포트폴리오 수익률
        title: 차트 제목
        
    Returns:
        Plotly Figure 객체
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly가 설치되지 않았습니다.")
    
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='드로우다운',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=1)
    ))
    
    # 0 라인 추가
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title='날짜',
        yaxis_title='드로우다운',
        yaxis_tickformat='.2%',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_returns_distribution_chart(
    returns: pd.Series,
    title: str = "수익률 분포"
) -> go.Figure:
    """수익률 분포 차트를 생성합니다.
    
    Args:
        returns: 포트폴리오 수익률
        title: 차트 제목
        
    Returns:
        Plotly Figure 객체
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly가 설치되지 않았습니다.")
    
    fig = go.Figure()
    
    # 히스토그램
    fig.add_trace(go.Histogram(
        x=returns.values,
        nbinsx=50,
        name='수익률 분포',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # 통계 라인들
    mean_return = returns.mean()
    std_return = returns.std()
    
    fig.add_vline(x=mean_return, line_dash="dash", line_color="red", 
                  annotation_text=f"평균: {mean_return:.4f}")
    fig.add_vline(x=mean_return + std_return, line_dash="dot", line_color="orange",
                  annotation_text=f"+1σ: {mean_return + std_return:.4f}")
    fig.add_vline(x=mean_return - std_return, line_dash="dot", line_color="orange",
                  annotation_text=f"-1σ: {mean_return - std_return:.4f}")
    
    fig.update_layout(
        title=title,
        xaxis_title='일일 수익률',
        yaxis_title='빈도',
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def create_rolling_metrics_chart(
    returns: pd.Series,
    window: int = 252,
    title: str = "롤링 지표"
) -> go.Figure:
    """롤링 지표 차트를 생성합니다.
    
    Args:
        returns: 포트폴리오 수익률
        window: 롤링 윈도우
        title: 차트 제목
        
    Returns:
        Plotly Figure 객체
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly가 설치되지 않았습니다.")
    
    # 롤링 지표 계산
    rolling_return = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['연간 수익률', '연간 변동성', '샤프 비율'],
        vertical_spacing=0.08
    )
    
    # 롤링 수익률
    fig.add_trace(go.Scatter(
        x=rolling_return.index,
        y=rolling_return.values,
        mode='lines',
        name='연간 수익률',
        line=dict(color='blue')
    ), row=1, col=1)
    
    # 롤링 변동성
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol.values,
        mode='lines',
        name='연간 변동성',
        line=dict(color='orange')
    ), row=2, col=1)
    
    # 롤링 샤프 비율
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode='lines',
        name='샤프 비율',
        line=dict(color='green')
    ), row=3, col=1)
    
    # 0 라인 추가 (샤프 비율)
    fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        showlegend=False,
        height=800
    )
    
    # Y축 포맷
    fig.update_yaxes(tickformat='.2%', row=1, col=1)
    fig.update_yaxes(tickformat='.2%', row=2, col=1)
    
    return fig


def create_strategy_comparison_chart(
    strategy_results: List[Dict[str, Any]],
    title: str = "전략 비교"
) -> go.Figure:
    """여러 전략의 성과를 비교하는 차트를 생성합니다.
    
    Args:
        strategy_results: 전략 결과 리스트
        title: 차트 제목
        
    Returns:
        Plotly Figure 객체
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly가 설치되지 않았습니다.")
    
    fig = go.Figure()
    
    for result in strategy_results:
        if 'portfolio_returns' in result:
            cumulative = (1 + result['portfolio_returns']).cumprod()
            strategy_name = result.get('strategy_name', 'Unknown')
            
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode='lines',
                name=strategy_name,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='날짜',
        yaxis_title='누적 수익률',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_performance_metrics_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    """성과 지표 테이블을 생성합니다.
    
    Args:
        metrics: 성과 지표 딕셔너리
        
    Returns:
        성과 지표 테이블
    """
    key_metrics = {
        '총 수익률': ('total_return', '{:.2%}'),
        '연간 수익률 (CAGR)': ('cagr', '{:.2%}'),
        '연간 변동성': ('volatility', '{:.2%}'),
        '샤프 비율': ('sharpe_ratio', '{:.3f}'),
        '소르티노 비율': ('sortino_ratio', '{:.3f}'),
        '최대 낙폭': ('max_drawdown', '{:.2%}'),
        '칼마 비율': ('calmar_ratio', '{:.3f}'),
        'VaR (95%)': ('var_95', '{:.2%}'),
        'CVaR (95%)': ('cvar_95', '{:.2%}'),
        '승률': ('win_rate', '{:.2%}'),
        '수익 팩터': ('profit_factor', '{:.3f}'),
        '최고 일일 수익률': ('best_day', '{:.2%}'),
        '최악 일일 수익률': ('worst_day', '{:.2%}')
    }
    
    table_data = []
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
            
            table_data.append({
                '지표': label,
                '값': formatted_value
            })
    
    return pd.DataFrame(table_data)


def generate_html_report(
    backtest_result: Dict[str, Any],
    output_path: Optional[Path] = None
) -> str:
    """HTML 리포트를 생성합니다.
    
    Args:
        backtest_result: 백테스트 결과
        output_path: 출력 파일 경로
        
    Returns:
        생성된 HTML 파일 경로
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly가 설치되지 않았습니다.")
    
    logger.info("📊 HTML 리포트 생성 시작")
    
    # 기본 정보
    strategy_name = backtest_result.get('strategy_name', 'Unknown')
    backtest_id = backtest_result.get('backtest_id', 'Unknown')
    
    if output_path is None:
        output_path = get_report_path(f"{strategy_name}_{backtest_id}.html")
    
    # 수익률 데이터
    returns = backtest_result.get('portfolio_returns')
    if returns is None:
        raise ValueError("포트폴리오 수익률 데이터가 없습니다.")
    
    # 차트 생성
    performance_chart = create_performance_chart(returns, title=f"{strategy_name} 성과")
    drawdown_chart = create_drawdown_chart(returns)
    distribution_chart = create_returns_distribution_chart(returns)
    rolling_chart = create_rolling_metrics_chart(returns)
    
    # 성과 지표 테이블
    metrics_table = create_performance_metrics_table(backtest_result)
    
    # HTML 템플릿
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{strategy_name} 백테스트 리포트</title>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f8f9fa;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            .metrics-table th, .metrics-table td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .metrics-table th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            .chart-container {{
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📈 {strategy_name} 백테스트 리포트</h1>
            <p><strong>백테스트 ID:</strong> {backtest_id}</p>
            <p><strong>생성 시간:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>기간:</strong> {backtest_result.get('start_date', 'N/A')} ~ {backtest_result.get('end_date', 'N/A')}</p>
        </div>
        
        <div class="section">
            <h2>📊 주요 성과 지표</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>지표</th>
                        <th>값</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # 성과 지표 테이블 추가
    for _, row in metrics_table.iterrows():
        html_template += f"""
                    <tr>
                        <td>{row['지표']}</td>
                        <td>{row['값']}</td>
                    </tr>
        """
    
    html_template += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>📈 누적 수익률</h2>
            <div class="chart-container" id="performance-chart"></div>
        </div>
        
        <div class="section">
            <h2>📉 드로우다운</h2>
            <div class="chart-container" id="drawdown-chart"></div>
        </div>
        
        <div class="section">
            <h2>📊 수익률 분포</h2>
            <div class="chart-container" id="distribution-chart"></div>
        </div>
        
        <div class="section">
            <h2>📈 롤링 지표</h2>
            <div class="chart-container" id="rolling-chart"></div>
        </div>
        
        <script>
    """
    
    # JavaScript로 차트 렌더링
    html_template += f"""
            Plotly.newPlot('performance-chart', {performance_chart.to_json()});
            Plotly.newPlot('drawdown-chart', {drawdown_chart.to_json()});
            Plotly.newPlot('distribution-chart', {distribution_chart.to_json()});
            Plotly.newPlot('rolling-chart', {rolling_chart.to_json()});
        </script>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    logger.success(f"✅ HTML 리포트 생성 완료: {output_path}")
    return str(output_path)


def generate_report(
    backtest_id: Optional[str] = None,
    latest: bool = False,
    output_path: Optional[str] = None
) -> str:
    """백테스트 결과 리포트를 생성합니다.
    
    Args:
        backtest_id: 백테스트 ID
        latest: 최신 결과 사용 여부
        output_path: 출력 파일 경로
        
    Returns:
        생성된 리포트 파일 경로
    """
    logger.info("📊 리포트 생성 시작")
    
    # 백테스트 결과 로드
    if latest:
        results_list = list_backtest_results()
        if not results_list:
            raise ValueError("저장된 백테스트 결과가 없습니다.")
        backtest_id = results_list[0]['backtest_id']  # 최신 결과
    
    if not backtest_id:
        raise ValueError("백테스트 ID가 필요합니다.")
    
    try:
        backtest_result = load_backtest_result(backtest_id)
    except FileNotFoundError:
        raise ValueError(f"백테스트 결과를 찾을 수 없습니다: {backtest_id}")
    
    # HTML 리포트 생성
    if output_path:
        output_path = Path(output_path)
    
    report_path = generate_html_report(backtest_result, output_path)
    
    logger.success(f"✅ 리포트 생성 완료: {report_path}")
    return report_path


# Dash 대시보드 (옵션)
def create_dashboard_app() -> 'dash.Dash':
    """Dash 대시보드 앱을 생성합니다.
    
    Returns:
        Dash 앱 인스턴스
    """
    if not DASH_AVAILABLE:
        raise ImportError("Dash가 설치되지 않았습니다.")
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # 백테스트 결과 목록
    results_list = list_backtest_results()
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("📈 Stock Forecast Lab 대시보드", className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("백테스트 결과 선택:"),
                dcc.Dropdown(
                    id='backtest-dropdown',
                    options=[
                        {
                            'label': f"{r['strategy_name']} ({r['backtest_id']})",
                            'value': r['backtest_id']
                        }
                        for r in results_list
                    ],
                    value=results_list[0]['backtest_id'] if results_list else None
                )
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='performance-graph')
            ], width=6),
            dbc.Col([
                dcc.Graph(id='drawdown-graph')
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='distribution-graph')
            ], width=6),
            dbc.Col([
                html.Div(id='metrics-table')
            ], width=6)
        ])
    ], fluid=True)
    
    @app.callback(
        [Output('performance-graph', 'figure'),
         Output('drawdown-graph', 'figure'),
         Output('distribution-graph', 'figure'),
         Output('metrics-table', 'children')],
        [Input('backtest-dropdown', 'value')]
    )
    def update_dashboard(backtest_id):
        if not backtest_id:
            return {}, {}, {}, "백테스트 결과를 선택하세요."
        
        try:
            result = load_backtest_result(backtest_id)
            returns = result.get('portfolio_returns')
            
            if returns is None:
                return {}, {}, {}, "수익률 데이터가 없습니다."
            
            # 차트 생성
            perf_fig = create_performance_chart(returns)
            dd_fig = create_drawdown_chart(returns)
            dist_fig = create_returns_distribution_chart(returns)
            
            # 지표 테이블
            metrics_table = create_performance_metrics_table(result)
            table_component = dbc.Table.from_dataframe(
                metrics_table, 
                striped=True, 
                bordered=True, 
                hover=True,
                size='sm'
            )
            
            return perf_fig, dd_fig, dist_fig, table_component
            
        except Exception as e:
            error_msg = f"오류: {str(e)}"
            return {}, {}, {}, error_msg
    
    return app


def start_dashboard(
    host: str = '127.0.0.1',
    port: int = 8050,
    debug: bool = True
):
    """대시보드를 시작합니다.
    
    Args:
        host: 호스트 주소
        port: 포트 번호
        debug: 디버그 모드
    """
    logger.info(f"🚀 대시보드 시작: http://{host}:{port}")
    
    try:
        app = create_dashboard_app()
        app.run_server(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"❌ 대시보드 시작 실패: {e}")


# CLI 직접 실행용
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python -m src.reports.plotly_dash report BACKTEST_ID")
        print("  python -m src.reports.plotly_dash dashboard")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "report":
        if len(sys.argv) < 3:
            print("백테스트 ID가 필요합니다.")
            sys.exit(1)
        
        bt_id = sys.argv[2]
        try:
            report_path = generate_report(backtest_id=bt_id)
            print(f"✅ 리포트 생성 완료: {report_path}")
        except Exception as e:
            print(f"❌ 리포트 생성 실패: {e}")
            sys.exit(1)
    
    elif command == "dashboard":
        try:
            start_dashboard()
        except KeyboardInterrupt:
            print("\n👋 대시보드를 종료합니다.")
        except Exception as e:
            print(f"❌ 대시보드 실행 실패: {e}")
            sys.exit(1)
    
    else:
        print(f"알 수 없는 명령어: {command}")
        sys.exit(1)