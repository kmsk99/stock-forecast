"""
Yahoo Finance 데이터 수집 모듈

yfinance 라이브러리를 사용하여 주식 데이터를 수집하고 저장합니다.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

import pandas as pd
import yfinance as yf
from loguru import logger

from ..config import settings
from ..utils.paths import get_raw_data_path, ensure_dir


warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")


def collect_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    save: bool = True
) -> Optional[pd.DataFrame]:
    """단일 종목 데이터를 수집합니다.
    
    Args:
        ticker: 종목 코드 (예: 'AAPL')
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        interval: 데이터 간격 ('1d', '1h', '5m' 등)
        save: 파일로 저장할지 여부
        
    Returns:
        수집된 데이터프레임 또는 None (실패시)
    """
    try:
        logger.info(f"📥 {ticker} 데이터 수집 시작: {start_date} ~ {end_date}")
        
        # yfinance 객체 생성
        stock = yf.Ticker(ticker)
        
        # 데이터 다운로드
        data = stock.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,  # 배당, 분할 조정
            prepost=False      # 시간외 거래 제외
        )
        
        if data.empty:
            logger.warning(f"⚠️ {ticker}: 데이터가 없습니다")
            return None
        
        # 컬럼명 정리
        data.columns = [col.replace(' ', '_').lower() for col in data.columns]
        
        # 인덱스를 컬럼으로 변환
        data = data.reset_index()
        
        # 인덱스 컬럼명 확인 및 처리
        index_col = data.columns[0]  # 첫 번째 컬럼이 날짜
        if index_col.lower() in ['date', 'datetime', 'timestamp']:
            data.rename(columns={index_col: 'date'}, inplace=True)
        else:
            data['date'] = data.index
        
        data['date'] = pd.to_datetime(data['date']).dt.date
        
        # 종목 코드 추가
        data['ticker'] = ticker
        
        # 컬럼 순서 정리
        columns_order = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        data = data[columns_order]
        
        # 결측치 제거
        data = data.dropna()
        
        if save:
            # 저장 경로 생성
            save_dir = get_raw_data_path() / start_date
            ensure_dir(save_dir)
            
            file_path = save_dir / f"{ticker}.csv"
            data.to_csv(file_path, index=False)
            logger.info(f"💾 {ticker} 데이터 저장 완료: {file_path}")
        
        logger.success(f"✅ {ticker} 수집 완료: {len(data)} 행")
        return data
        
    except Exception as e:
        logger.error(f"❌ {ticker} 수집 실패: {e}")
        return None


def collect_multiple_tickers(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    save: bool = True
) -> Dict[str, pd.DataFrame]:
    """여러 종목 데이터를 수집합니다.
    
    Args:
        tickers: 종목 코드 리스트
        start_date: 시작 날짜
        end_date: 종료 날짜
        interval: 데이터 간격
        save: 파일로 저장할지 여부
        
    Returns:
        종목별 데이터프레임 딕셔너리
    """
    logger.info(f"📊 여러 종목 데이터 수집 시작: {len(tickers)} 종목")
    
    results = {}
    success_count = 0
    
    for ticker in tickers:
        data = collect_single_ticker(ticker, start_date, end_date, interval, save)
        if data is not None:
            results[ticker] = data
            success_count += 1
    
    logger.info(f"🎯 수집 완료: {success_count}/{len(tickers)} 종목 성공")
    return results


def collect_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """메인 데이터 수집 함수 (CLI에서 호출)
    
    Args:
        tickers: 종목 코드 리스트
        start_date: 시작 날짜
        end_date: 종료 날짜
        interval: 데이터 간격
        
    Returns:
        수집된 데이터 딕셔너리
    """
    return collect_multiple_tickers(tickers, start_date, end_date, interval, save=True)


def get_stock_info(ticker: str) -> Optional[Dict[str, Any]]:
    """종목 기본 정보를 가져옵니다.
    
    Args:
        ticker: 종목 코드
        
    Returns:
        종목 정보 딕셔너리
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 주요 정보만 추출
        key_info = {
            'symbol': info.get('symbol', ticker),
            'shortName': info.get('shortName', ''),
            'longName': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'marketCap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', ''),
            'country': info.get('country', ''),
        }
        
        return key_info
        
    except Exception as e:
        logger.error(f"❌ {ticker} 정보 조회 실패: {e}")
        return None


def validate_tickers(tickers: List[str]) -> List[str]:
    """유효한 종목 코드만 필터링합니다.
    
    Args:
        tickers: 검증할 종목 코드 리스트
        
    Returns:
        유효한 종목 코드 리스트
    """
    valid_tickers = []
    
    logger.info(f"🔍 종목 코드 검증 시작: {len(tickers)} 종목")
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # 최근 1일 데이터로 유효성 검사
            test_data = stock.history(period="1d")
            
            if not test_data.empty:
                valid_tickers.append(ticker)
                logger.debug(f"✅ {ticker}: 유효")
            else:
                logger.warning(f"⚠️ {ticker}: 데이터 없음")
                
        except Exception as e:
            logger.warning(f"⚠️ {ticker}: 검증 실패 - {e}")
    
    logger.info(f"🎯 검증 완료: {len(valid_tickers)}/{len(tickers)} 종목 유효")
    return valid_tickers


def update_existing_data(
    ticker: str,
    last_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """기존 데이터를 업데이트합니다.
    
    Args:
        ticker: 종목 코드
        last_date: 마지막 데이터 날짜 (없으면 자동 감지)
        
    Returns:
        업데이트된 데이터
    """
    try:
        # 기존 파일 찾기
        raw_dir = get_raw_data_path()
        existing_files = list(raw_dir.rglob(f"{ticker}.csv"))
        
        if not existing_files:
            logger.warning(f"⚠️ {ticker}: 기존 데이터가 없습니다")
            return None
        
        # 가장 최근 파일 찾기
        latest_file = max(existing_files, key=lambda x: x.parent.name)
        existing_data = pd.read_csv(latest_file)
        
        if last_date is None:
            # 마지막 날짜 자동 감지
            existing_data['date'] = pd.to_datetime(existing_data['date'])
            last_date = existing_data['date'].max().strftime('%Y-%m-%d')
        
        # 다음 날부터 오늘까지 수집
        from datetime import datetime, timedelta
        next_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')
        
        if next_date >= today:
            logger.info(f"🎯 {ticker}: 이미 최신 데이터입니다")
            return existing_data
        
        # 새 데이터 수집
        new_data = collect_single_ticker(ticker, next_date, today, save=False)
        
        if new_data is not None and not new_data.empty:
            # 데이터 병합
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=['date', 'ticker'])
            combined_data = combined_data.sort_values(['date'])
            
            # 파일 덮어쓰기
            combined_data.to_csv(latest_file, index=False)
            logger.success(f"✅ {ticker} 데이터 업데이트 완료: +{len(new_data)} 행")
            
            return combined_data
        else:
            logger.info(f"🎯 {ticker}: 새로운 데이터가 없습니다")
            return existing_data
            
    except Exception as e:
        logger.error(f"❌ {ticker} 업데이트 실패: {e}")
        return None


# CLI 직접 실행용
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("사용법: python -m src.ingest.yfinance_cli TICKER START_DATE END_DATE")
        sys.exit(1)
    
    ticker = sys.argv[1]
    start = sys.argv[2] 
    end = sys.argv[3]
    
    data = collect_single_ticker(ticker, start, end)
    if data is not None:
        print(f"✅ {ticker} 데이터 수집 완료: {len(data)} 행")
    else:
        print(f"❌ {ticker} 데이터 수집 실패")
        sys.exit(1)