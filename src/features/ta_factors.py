"""
기술지표 (Technical Analysis) 피처 생성 모듈

다양한 기술지표를 계산하여 피처로 변환합니다.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
import polars as pl
from loguru import logger

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    logger.warning("ta 라이브러리가 설치되지 않았습니다. 기본 지표만 사용됩니다.")
    TA_AVAILABLE = False

from ..utils.paths import get_raw_data_path, get_silver_data_path, ensure_dir


def calculate_sma(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """단순 이동평균을 계산합니다.
    
    Args:
        data: 가격 데이터
        window: 이동평균 기간
        
    Returns:
        SMA 시리즈
    """
    return data['close'].rolling(window=window).mean()


def calculate_ema(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """지수 이동평균을 계산합니다.
    
    Args:
        data: 가격 데이터
        window: 이동평균 기간
        
    Returns:
        EMA 시리즈
    """
    return data['close'].ewm(span=window).mean()


def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """RSI (Relative Strength Index)를 계산합니다.
    
    Args:
        data: 가격 데이터
        window: RSI 기간
        
    Returns:
        RSI 시리즈
    """
    if TA_AVAILABLE:
        return ta.momentum.RSIIndicator(data['close'], window=window).rsi()
    
    # 수동 RSI 계산
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """MACD를 계산합니다.
    
    Args:
        data: 가격 데이터
        fast: 빠른 EMA 기간
        slow: 느린 EMA 기간
        signal: 시그널 라인 기간
        
    Returns:
        MACD, Signal, Histogram 딕셔너리
    """
    if TA_AVAILABLE:
        macd_indicator = ta.trend.MACD(data['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        return {
            'macd': macd_indicator.macd(),
            'macd_signal': macd_indicator.macd_signal(),
            'macd_histogram': macd_indicator.macd_diff()
        }
    
    # 수동 MACD 계산
    ema_fast = data['close'].ewm(span=fast).mean()
    ema_slow = data['close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    
    return {
        'macd': macd,
        'macd_signal': macd_signal,
        'macd_histogram': macd_histogram
    }


def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
    """볼린저 밴드를 계산합니다.
    
    Args:
        data: 가격 데이터
        window: 이동평균 기간
        std_dev: 표준편차 배수
        
    Returns:
        Upper, Middle, Lower 밴드 딕셔너리
    """
    if TA_AVAILABLE:
        bb = ta.volatility.BollingerBands(data['close'], window=window, window_dev=std_dev)
        return {
            'bb_upper': bb.bollinger_hband(),
            'bb_middle': bb.bollinger_mavg(),
            'bb_lower': bb.bollinger_lband()
        }
    
    # 수동 볼린저 밴드 계산
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    
    return {
        'bb_upper': sma + (std * std_dev),
        'bb_middle': sma,
        'bb_lower': sma - (std * std_dev)
    }


def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR (Average True Range)를 계산합니다.
    
    Args:
        data: OHLC 데이터
        window: ATR 기간
        
    Returns:
        ATR 시리즈
    """
    if TA_AVAILABLE:
        return ta.volatility.AverageTrueRange(
            data['high'], data['low'], data['close'], window=window
        ).average_true_range()
    
    # 수동 ATR 계산
    high_low = data['high'] - data['low']
    high_close_prev = np.abs(data['high'] - data['close'].shift(1))
    low_close_prev = np.abs(data['low'] - data['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr


def calculate_stochastic(data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
    """스토캐스틱 오실레이터를 계산합니다.
    
    Args:
        data: OHLC 데이터
        k_window: %K 기간
        d_window: %D 기간
        
    Returns:
        %K, %D 딕셔너리
    """
    if TA_AVAILABLE:
        stoch = ta.momentum.StochasticOscillator(
            data['high'], data['low'], data['close'], window=k_window, smooth_window=d_window
        )
        return {
            'stoch_k': stoch.stoch(),
            'stoch_d': stoch.stoch_signal()
        }
    
    # 수동 스토캐스틱 계산
    lowest_low = data['low'].rolling(window=k_window).min()
    highest_high = data['high'].rolling(window=k_window).max()
    
    k_percent = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return {
        'stoch_k': k_percent,
        'stoch_d': d_percent
    }


def calculate_volume_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """거래량 지표들을 계산합니다.
    
    Args:
        data: OHLCV 데이터
        
    Returns:
        거래량 지표 딕셔너리
    """
    indicators = {}
    
    # 거래량 이동평균
    indicators['volume_sma_20'] = data['volume'].rolling(window=20).mean()
    indicators['volume_sma_50'] = data['volume'].rolling(window=50).mean()
    
    # 거래량 비율
    indicators['volume_ratio'] = data['volume'] / indicators['volume_sma_20']
    
    # 가격 변화율과 거래량의 관계
    price_change = data['close'].pct_change()
    indicators['price_volume_trend'] = (price_change * data['volume']).cumsum()
    
    if TA_AVAILABLE:
        # On-Balance Volume
        indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(
            data['close'], data['volume']
        ).on_balance_volume()
        
        # Volume Weighted Average Price (근사치)
        indicators['vwap'] = (data['close'] * data['volume']).rolling(window=20).sum() / \
                           data['volume'].rolling(window=20).sum()
    
    return indicators


def calculate_price_features(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """가격 기반 피처들을 계산합니다.
    
    Args:
        data: OHLC 데이터
        
    Returns:
        가격 피처 딕셔너리
    """
    features = {}
    
    # 수익률
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # 변동성 (Rolling)
    features['volatility_10'] = features['returns'].rolling(window=10).std()
    features['volatility_20'] = features['returns'].rolling(window=20).std()
    
    # 가격 차이
    features['high_low_pct'] = (data['high'] - data['low']) / data['close']
    features['open_close_pct'] = (data['close'] - data['open']) / data['open']
    
    # 갭
    features['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # 상대적 위치
    features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    return features


def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """모든 기술지표를 추가합니다.
    
    Args:
        data: 기본 OHLCV 데이터
        
    Returns:
        기술지표가 추가된 데이터프레임
    """
    logger.info(f"📊 기술지표 계산 시작: {len(data)} 행")
    
    # 복사본 생성
    result = data.copy()
    
    # 이동평균
    result['sma_5'] = calculate_sma(data, 5)
    result['sma_10'] = calculate_sma(data, 10)
    result['sma_20'] = calculate_sma(data, 20)
    result['sma_50'] = calculate_sma(data, 50)
    
    result['ema_5'] = calculate_ema(data, 5)
    result['ema_10'] = calculate_ema(data, 10)
    result['ema_20'] = calculate_ema(data, 20)
    
    # 모멘텀 지표
    result['rsi'] = calculate_rsi(data)
    
    # MACD
    macd_dict = calculate_macd(data)
    for key, value in macd_dict.items():
        result[key] = value
    
    # 볼린저 밴드
    bb_dict = calculate_bollinger_bands(data)
    for key, value in bb_dict.items():
        result[key] = value
    
    # 변동성 지표
    result['atr'] = calculate_atr(data)
    
    # 스토캐스틱
    stoch_dict = calculate_stochastic(data)
    for key, value in stoch_dict.items():
        result[key] = value
    
    # 거래량 지표
    volume_dict = calculate_volume_indicators(data)
    for key, value in volume_dict.items():
        result[key] = value
    
    # 가격 피처
    price_dict = calculate_price_features(data)
    for key, value in price_dict.items():
        result[key] = value
    
    logger.success(f"✅ 기술지표 계산 완료: {len(result.columns)} 컬럼")
    return result


def process_single_ticker(ticker_data: pd.DataFrame) -> pd.DataFrame:
    """단일 종목 데이터에 대해 피처를 생성합니다.
    
    Args:
        ticker_data: 종목 데이터
        
    Returns:
        피처가 추가된 데이터프레임
    """
    # 날짜 정렬
    ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
    
    # 기술지표 추가
    enhanced_data = add_all_indicators(ticker_data)
    
    # 결측치가 너무 많은 초기 행들 제거
    enhanced_data = enhanced_data.dropna(thresh=len(enhanced_data.columns) * 0.7)
    
    return enhanced_data


def create_features(
    input_dir: Path,
    output_dir: Path,
    force: bool = False
) -> str:
    """원본 데이터에서 피처를 생성하여 저장합니다.
    
    Args:
        input_dir: 원본 데이터 디렉토리
        output_dir: 출력 디렉토리
        force: 기존 파일 덮어쓰기 여부
        
    Returns:
        결과 메시지
    """
    logger.info(f"🔧 피처 생성 시작: {input_dir} -> {output_dir}")
    
    # 출력 디렉토리 생성
    ensure_dir(output_dir)
    
    # 원본 CSV 파일들 찾기
    csv_files = list(input_dir.rglob("*.csv"))
    
    if not csv_files:
        return "원본 데이터 파일을 찾을 수 없습니다."
    
    logger.info(f"📁 발견된 파일: {len(csv_files)} 개")
    
    all_features = []
    processed_count = 0
    
    for csv_file in csv_files:
        try:
            # 파일 읽기
            data = pd.read_csv(csv_file)
            data['date'] = pd.to_datetime(data['date'])
            
            # 피처 생성
            features = process_single_ticker(data)
            
            if not features.empty:
                all_features.append(features)
                processed_count += 1
                logger.debug(f"✅ {csv_file.name} 처리 완료")
            
        except Exception as e:
            logger.error(f"❌ {csv_file.name} 처리 실패: {e}")
            continue
    
    if not all_features:
        return "처리 가능한 데이터가 없습니다."
    
    # 모든 데이터 합치기
    combined_features = pd.concat(all_features, ignore_index=True)
    
    # MultiIndex 생성 (날짜, 종목)
    combined_features = combined_features.set_index(['date', 'ticker']).sort_index()
    
    # Parquet으로 저장
    output_file = output_dir / "features.parquet"
    
    if output_file.exists() and not force:
        return f"파일이 이미 존재합니다: {output_file} (--force 옵션 사용)"
    
    combined_features.to_parquet(output_file)
    
    logger.success(f"✅ 피처 생성 완료: {processed_count} 종목, {len(combined_features)} 행")
    return f"피처 파일 생성 완료: {output_file}"


def load_features(file_path: Optional[Path] = None) -> pd.DataFrame:
    """저장된 피처 데이터를 로드합니다.
    
    Args:
        file_path: 피처 파일 경로 (None이면 기본 경로)
        
    Returns:
        피처 데이터프레임
    """
    if file_path is None:
        file_path = get_silver_data_path("features.parquet")
    
    if not file_path.exists():
        raise FileNotFoundError(f"피처 파일을 찾을 수 없습니다: {file_path}")
    
    return pd.read_parquet(file_path)


# CLI 직접 실행용
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("사용법: python -m src.features.ta_factors INPUT_DIR OUTPUT_DIR")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    result = create_features(input_path, output_path)
    print(result)