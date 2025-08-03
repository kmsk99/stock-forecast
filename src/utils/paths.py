"""
경로 관리 유틸리티 모듈

프로젝트 내 경로 관련 헬퍼 함수들을 제공합니다.
"""

import os
from pathlib import Path
from typing import Optional, Union

from ..config import settings


def get_project_root() -> Path:
    """프로젝트 루트 경로를 반환합니다."""
    return settings.project_root


def get_data_path(subpath: Optional[str] = None) -> Path:
    """데이터 디렉토리 경로를 반환합니다.
    
    Args:
        subpath: 데이터 디렉토리 내 하위 경로
        
    Returns:
        데이터 디렉토리 경로
    """
    if subpath:
        return settings.data_dir / subpath
    return settings.data_dir


def get_raw_data_path(ticker: Optional[str] = None, date: Optional[str] = None) -> Path:
    """원본 데이터 경로를 반환합니다.
    
    Args:
        ticker: 종목 코드 (예: 'AAPL')
        date: 날짜 (예: '2024-01-01')
        
    Returns:
        원본 데이터 경로
    """
    path = settings.raw_data_dir
    
    if date:
        path = path / date
    
    if ticker:
        path = path / f"{ticker}.csv"
    
    return path


def get_silver_data_path(filename: Optional[str] = None) -> Path:
    """가공 데이터 경로를 반환합니다.
    
    Args:
        filename: 파일명 (예: 'features.parquet')
        
    Returns:
        가공 데이터 경로
    """
    path = settings.silver_data_dir
    
    if filename:
        path = path / filename
        
    return path


def get_cache_path(filename: Optional[str] = None) -> Path:
    """캐시 디렉토리 경로를 반환합니다.
    
    Args:
        filename: 캐시 파일명
        
    Returns:
        캐시 경로
    """
    path = settings.cache_dir
    
    if filename:
        path = path / filename
        
    return path


def get_log_path() -> Path:
    """로그 파일 경로를 반환합니다."""
    return settings.log_file


def ensure_dir(path: Union[str, Path]) -> Path:
    """디렉토리가 존재하지 않으면 생성합니다.
    
    Args:
        path: 생성할 디렉토리 경로
        
    Returns:
        생성된 디렉토리 경로
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_backtest_result_path(bt_id: Optional[str] = None) -> Path:
    """백테스트 결과 저장 경로를 반환합니다.
    
    Args:
        bt_id: 백테스트 ID (예: '20241231T1230')
        
    Returns:
        백테스트 결과 경로
    """
    results_dir = get_data_path("backtest_results")
    ensure_dir(results_dir)
    
    if bt_id:
        return results_dir / f"{bt_id}.pkl"
    
    return results_dir


def get_report_path(report_id: Optional[str] = None) -> Path:
    """리포트 저장 경로를 반환합니다.
    
    Args:
        report_id: 리포트 ID
        
    Returns:
        리포트 경로
    """
    reports_dir = get_data_path("reports")
    ensure_dir(reports_dir)
    
    if report_id:
        return reports_dir / f"{report_id}.html"
    
    return reports_dir


def list_raw_data_files() -> list[Path]:
    """원본 데이터 파일 목록을 반환합니다."""
    raw_dir = get_raw_data_path()
    
    if not raw_dir.exists():
        return []
    
    files = []
    for item in raw_dir.rglob("*.csv"):
        files.append(item)
    
    return sorted(files)


def list_silver_data_files() -> list[Path]:
    """가공 데이터 파일 목록을 반환합니다."""
    silver_dir = get_silver_data_path()
    
    if not silver_dir.exists():
        return []
    
    files = []
    for pattern in ["*.parquet", "*.feather", "*.csv"]:
        files.extend(silver_dir.glob(pattern))
    
    return sorted(files)


def get_temp_path(filename: Optional[str] = None) -> Path:
    """임시 파일 경로를 반환합니다.
    
    Args:
        filename: 임시 파일명
        
    Returns:
        임시 파일 경로
    """
    temp_dir = get_cache_path("temp")
    ensure_dir(temp_dir)
    
    if filename:
        return temp_dir / filename
    
    return temp_dir


def clean_cache(older_than_hours: Optional[int] = None) -> int:
    """캐시를 정리합니다.
    
    Args:
        older_than_hours: 지정한 시간보다 오래된 파일만 삭제
        
    Returns: 
        삭제된 파일 수
    """
    cache_dir = get_cache_path()
    
    if not cache_dir.exists():
        return 0
    
    import time
    from datetime import datetime, timedelta
    
    deleted_count = 0
    cutoff_time = None
    
    if older_than_hours:
        cutoff_time = time.time() - (older_than_hours * 3600)
    
    for file_path in cache_dir.rglob("*"):
        if file_path.is_file():
            if cutoff_time is None or file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except OSError:
                    continue
    
    return deleted_count