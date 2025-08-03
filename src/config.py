"""
Stock Forecast Lab 설정 관리 모듈

Pydantic BaseSettings를 사용하여 환경변수와 설정을 중앙 관리합니다.
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정 클래스"""
    
    # 프로젝트 경로 설정
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="프로젝트 루트 경로"
    )
    
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data",
        description="데이터 디렉토리 경로"
    )
    
    raw_data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "raw",
        description="원본 데이터 디렉토리 경로"
    )
    
    silver_data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "silver",
        description="가공 데이터 디렉토리 경로"
    )
    
    # API 키 설정
    yahoo_finance_api_key: Optional[str] = Field(
        default=None,
        description="Yahoo Finance API 키 (현재는 불필요)"
    )
    
    alpha_vantage_api_key: Optional[str] = Field(
        default=None,
        description="Alpha Vantage API 키"
    )
    
    quandl_api_key: Optional[str] = Field(
        default=None,
        description="Quandl API 키"
    )
    
    dart_api_key: Optional[str] = Field(
        default=None,
        description="DART API 키 (한국 공시정보)"
    )
    
    # 데이터베이스 설정
    database_url: str = Field(
        default="sqlite:///stock_data.db",
        description="데이터베이스 연결 URL"
    )
    
    # 로깅 설정
    log_level: str = Field(
        default="INFO",
        description="로그 레벨"
    )
    
    log_format: str = Field(
        default="json",
        description="로그 형식 (json, text)"
    )
    
    log_file: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "logs" / "stocklab.log",
        description="로그 파일 경로"
    )
    
    # 백테스트 기본 설정
    default_start_date: str = Field(
        default="2020-01-01",
        description="기본 시작 날짜"
    )
    
    default_end_date: str = Field(
        default="2024-12-31",
        description="기본 종료 날짜"
    )
    
    default_rebalance_freq: str = Field(
        default="1M",
        description="기본 리밸런싱 주기"
    )
    
    default_transaction_cost: float = Field(
        default=0.0005,
        description="기본 거래 비용 (0.05%)"
    )
    
    default_slippage: float = Field(
        default=0.0002,
        description="기본 슬리피지 (0.02%)"
    )
    
    # 대시보드 설정
    dash_host: str = Field(
        default="127.0.0.1",
        description="Dash 호스트"
    )
    
    dash_port: int = Field(
        default=8050,
        description="Dash 포트"
    )
    
    dash_debug: bool = Field(
        default=True,
        description="Dash 디버그 모드"
    )
    
    # 캐시 설정
    cache_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / ".cache",
        description="캐시 디렉토리"
    )
    
    cache_expire_hours: int = Field(
        default=24,
        description="캐시 만료 시간 (시간)"
    )
    
    # 병렬 처리 설정
    n_jobs: int = Field(
        default=4,
        description="병렬 처리 작업 수"
    )
    
    chunk_size: int = Field(
        default=1000,
        description="청크 크기"
    )
    
    # 알림 설정 (옵션)
    slack_webhook_url: Optional[str] = Field(
        default=None,
        description="Slack 웹훅 URL"
    )
    
    discord_webhook_url: Optional[str] = Field(
        default=None,
        description="Discord 웹훅 URL"
    )
    
    email_smtp_server: Optional[str] = Field(
        default=None,
        description="이메일 SMTP 서버"
    )
    
    email_smtp_port: int = Field(
        default=587,
        description="이메일 SMTP 포트"
    )
    
    email_username: Optional[str] = Field(
        default=None,
        description="이메일 사용자명"
    )
    
    email_password: Optional[str] = Field(
        default=None,
        description="이메일 비밀번호"
    )
    
    class Config:
        """Pydantic 설정"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def model_post_init(self, __context) -> None:
        """모델 초기화 후 디렉토리 생성"""
        # 필요한 디렉토리 생성
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.silver_data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


# 전역 설정 인스턴스
settings = Settings()


def get_settings() -> Settings:
    """설정 인스턴스를 반환합니다."""
    return settings


# 자주 사용하는 경로들을 편의상 export
PROJECT_ROOT = settings.project_root
DATA_DIR = settings.data_dir
RAW_DATA_DIR = settings.raw_data_dir
SILVER_DATA_DIR = settings.silver_data_dir
CACHE_DIR = settings.cache_dir