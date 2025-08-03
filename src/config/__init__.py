"""
Configuration management module
"""

# settings를 기존 위치에서 import
import sys
from pathlib import Path

# src 디렉토리를 Python path에 추가
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

try:
    from config import settings
except ImportError:
    # 대체 import 경로
    class MockSettings:
        # 경로 설정
        project_root = Path.cwd()
        data_dir = Path("data")
        raw_data_dir = Path("data/raw") 
        silver_data_dir = Path("data/silver")
        cache_dir = Path("cache")
        reports_dir = Path("reports")
        log_file = Path("logs/stocklab.log")
        
        # API 키들
        alpha_vantage_api_key = None
        quandl_api_key = None
        fred_api_key = None
        dart_api_key = None
        
        # 데이터베이스
        database_url = "sqlite:///stock_data.db"
        
        # 로깅
        log_level = "INFO"
        log_format = "json"
        
        # 백테스트
        default_start_date = "2020-01-01"
        default_end_date = "2024-12-31"
        default_rebalance_freq = "1M"
        default_transaction_cost = 0.0005
        default_slippage = 0.0002
        
        # 대시보드
        dash_host = "127.0.0.1"
        dash_port = 8050
        dash_debug = True
        
        # 위 속성들의 대문자 버전도 지원 (호환성)
        DATA_DIR = data_dir
        DATA_RAW_DIR = raw_data_dir
        DATA_SILVER_DIR = silver_data_dir
        REPORTS_DIR = reports_dir
    
    settings = MockSettings()