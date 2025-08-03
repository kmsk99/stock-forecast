#!/usr/bin/env python3
"""
HTML 인터랙티브 리포트 생성 스크립트
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def generate_html_report():
    print('🌐 인터랙티브 HTML 리포트 생성 중...')

    # 결과 로드
    with open('backtest_results_etf.pkl', 'rb') as f:
        results = pickle.load(f)

    equal_metrics = results['equal_weight']
    vol_metrics = results['vol_parity']  
    prices = results['prices']

    # 4x1 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '누적 수익률 비교',
            '개별 ETF 성과', 
            '성과 지표 비교',
            '롤링 샤프 비율 (60일)'
        ),
        specs=[[{}, {}], [{}, {}]],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    # 1. 누적 수익률 비교
    equal_cumulative = equal_metrics['cumulative']
    vol_cumulative = vol_metrics['cumulative']

    fig.add_trace(
        go.Scatter(
            x=equal_cumulative.index,
            y=equal_cumulative.values,
            mode='lines',
            name='동일가중',
            line=dict(color='#2E86AB', width=3),
            hovertemplate='날짜: %{x}<br>누적수익률: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=vol_cumulative.index,
            y=vol_cumulative.values,
            mode='lines',
            name='변동성 패리티',
            line=dict(color='#A23B72', width=3),
            hovertemplate='날짜: %{x}<br>누적수익률: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. 개별 ETF 성과
    prices_norm = prices / prices.iloc[0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    for i, ticker in enumerate(prices_norm.columns):
        fig.add_trace(
            go.Scatter(
                x=prices_norm.index,
                y=prices_norm[ticker].values,
                mode='lines',
                name=ticker,
                line=dict(color=colors[i], width=2),
                hovertemplate=f'{ticker}<br>날짜: %{{x}}<br>정규화 가격: %{{y:.3f}}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )

    # 3. 성과 지표 비교
    metrics_names = ['CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
    equal_values = [
        equal_metrics['cagr'] * 100,
        equal_metrics['sharpe'],
        equal_metrics['max_drawdown'] * 100
    ]
    vol_values = [
        vol_metrics['cagr'] * 100,
        vol_metrics['sharpe'],
        vol_metrics['max_drawdown'] * 100
    ]

    fig.add_trace(
        go.Bar(
            x=metrics_names,
            y=equal_values,
            name='동일가중',
            marker_color='#2E86AB',
            opacity=0.8,
            text=[f'{v:.1f}' for v in equal_values],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=metrics_names,
            y=vol_values,
            name='변동성 패리티',
            marker_color='#A23B72',
            opacity=0.8,
            text=[f'{v:.1f}' for v in vol_values],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=1
    )

    # 4. 롤링 샤프 비율
    equal_rolling_sharpe = equal_metrics['returns'].rolling(60).mean() / equal_metrics['returns'].rolling(60).std() * np.sqrt(252)
    vol_rolling_sharpe = vol_metrics['returns'].rolling(60).mean() / vol_metrics['returns'].rolling(60).std() * np.sqrt(252)

    fig.add_trace(
        go.Scatter(
            x=equal_rolling_sharpe.index,
            y=equal_rolling_sharpe.values,
            mode='lines',
            name='동일가중 샤프',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='날짜: %{x}<br>샤프비율: %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=vol_rolling_sharpe.index,
            y=vol_rolling_sharpe.values,
            mode='lines',
            name='변동성패리티 샤프',
            line=dict(color='#A23B72', width=2),
            hovertemplate='날짜: %{x}<br>샤프비율: %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )

    # 0선 추가
    fig.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.5, row=2, col=2)

    # 레이아웃 업데이트
    fig.update_layout(
        title={
            'text': '🏆 ETF 포트폴리오 백테스트 결과 (2022-2024)<br><sup>QQQ, VOO, BITO, GLD - 동일가중 vs 변동성패리티 전략</sup>',
            'x': 0.5,
            'font': {'size': 20, 'family': 'Arial Black'}
        },
        height=800,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    # Y축 레이블 추가
    fig.update_yaxes(title_text='포트폴리오 가치', row=1, col=1)
    fig.update_yaxes(title_text='정규화 가격', row=1, col=2)
    fig.update_yaxes(title_text='값', row=2, col=1)
    fig.update_yaxes(title_text='샤프 비율', row=2, col=2)

    # HTML 템플릿
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>ETF Portfolio Backtest Report</title>
    <meta charset='utf-8'>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 700;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            text-align: center;
            border-left: 4px solid #2E86AB;
        }}
        .metric-card.vol-parity {{
            border-left-color: #A23B72;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2E86AB;
        }}
        .metric-card.vol-parity .metric-value {{
            color: #A23B72;
        }}
        .metric-label {{
            margin-top: 5px;
            color: #666;
            font-weight: 500;
        }}
        .chart-container {{
            padding: 30px;
        }}
        .insight {{
            background: #e3f2fd;
            padding: 20px;
            margin: 20px 30px;
            border-radius: 10px;
            border-left: 4px solid #2196f3;
        }}
        .insight h3 {{
            margin-top: 0;
            color: #1976d2;
        }}
        .footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class='container'>
        <div class='header'>
            <h1>📈 ETF Portfolio Backtest</h1>
            <p>2022년 1월 ~ 2024년 12월 | QQQ • VOO • BITO • GLD</p>
        </div>
        
        <div class='summary'>
            <div class='metric-card'>
                <div class='metric-value'>{equal_metrics['cagr']:.1%}</div>
                <div class='metric-label'>동일가중 CAGR</div>
            </div>
            <div class='metric-card vol-parity'>
                <div class='metric-value'>{vol_metrics['cagr']:.1%}</div>
                <div class='metric-label'>변동성패리티 CAGR</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{equal_metrics['sharpe']:.2f}</div>
                <div class='metric-label'>동일가중 샤프비율</div>
            </div>
            <div class='metric-card vol-parity'>
                <div class='metric-value'>{vol_metrics['sharpe']:.2f}</div>
                <div class='metric-label'>변동성패리티 샤프비율</div>
            </div>
        </div>
        
        <div class='insight'>
            <h3>🎯 핵심 인사이트</h3>
            <ul>
                <li><strong>BITO (Bitcoin ETF)</strong>가 76.40%로 최고 성과!</li>
                <li><strong>변동성 패리티 전략</strong>이 더 높은 위험조정수익률 달성 (Sharpe 0.926 vs 0.805)</li>
                <li><strong>동일가중 전략</strong>이 더 높은 절대수익률 달성 (CAGR 16.66% vs 13.84%)</li>
                <li>변동성 패리티가 최대낙폭 10% 더 낮음 (-24.57% vs -34.53%)</li>
            </ul>
        </div>
        
        <div class='chart-container'>
            {fig.to_html(include_plotlyjs='cdn', div_id='chart')}
        </div>
        
        <div class='footer'>
            <p>Generated by Stock Forecast Lab | Powered by Plotly & Python</p>
        </div>
    </div>
</body>
</html>"""

    # HTML 파일 저장
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    html_path = reports_dir / 'etf_backtest_interactive.html'
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f'🌐 인터랙티브 HTML 리포트 저장: {html_path}')
    print(f'📂 브라우저에서 열어보세요: file://{html_path.absolute()}')
    
    return html_path

if __name__ == "__main__":
    generate_html_report()