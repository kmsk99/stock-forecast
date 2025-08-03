"""
레이블 생성 모듈

미래 수익률을 기반으로 회귀 및 분류 레이블을 생성합니다.
"""

from typing import List, Optional, Dict, Any, Tuple
import warnings

import pandas as pd
import numpy as np
from loguru import logger


def calculate_forward_returns(
    data: pd.DataFrame,
    periods: List[int] = [1, 5, 10, 20],
    price_col: str = 'close'
) -> pd.DataFrame:
    """미래 수익률을 계산합니다.
    
    Args:
        data: 가격 데이터 (종목별로 정렬되어야 함)
        periods: 예측 기간 리스트 (일 단위)
        price_col: 가격 컬럼명
        
    Returns:
        미래 수익률이 추가된 데이터프레임
    """
    result = data.copy()
    
    for period in periods:
        # 미래 가격
        future_price = data[price_col].shift(-period)
        
        # 수익률 계산
        forward_return = (future_price - data[price_col]) / data[price_col]
        
        # 로그 수익률
        forward_log_return = np.log(future_price / data[price_col])
        
        result[f'forward_return_{period}d'] = forward_return
        result[f'forward_log_return_{period}d'] = forward_log_return
    
    return result


def create_classification_labels(
    data: pd.DataFrame,
    return_col: str,
    method: str = 'quantile',
    n_classes: int = 3,
    thresholds: Optional[List[float]] = None
) -> pd.Series:
    """분류 레이블을 생성합니다.
    
    Args:
        data: 데이터프레임
        return_col: 수익률 컬럼명
        method: 분류 방법 ('quantile', 'threshold', 'zscore')
        n_classes: 클래스 수 (quantile 방법용)
        thresholds: 임계값 리스트 (threshold 방법용)
        
    Returns:
        분류 레이블 시리즈
    """
    returns = data[return_col].dropna()
    
    if method == 'quantile':
        # 분위수 기반 분류
        labels = pd.qcut(
            returns, 
            q=n_classes, 
            labels=list(range(n_classes)),
            duplicates='drop'
        )
        
    elif method == 'threshold':
        # 임계값 기반 분류
        if thresholds is None:
            thresholds = [-0.02, 0.02]  # 기본값: -2%, +2%
        
        labels = pd.cut(
            returns,
            bins=[-np.inf] + thresholds + [np.inf],
            labels=list(range(len(thresholds) + 1))
        )
        
    elif method == 'zscore':
        # Z-스코어 기반 분류
        z_scores = (returns - returns.mean()) / returns.std()
        
        if thresholds is None:
            thresholds = [-1.0, 1.0]  # 기본값: ±1 표준편차
        
        labels = pd.cut(
            z_scores,
            bins=[-np.inf] + thresholds + [np.inf],
            labels=list(range(len(thresholds) + 1))
        )
        
    else:
        raise ValueError(f"지원하지 않는 방법: {method}")
    
    # 원본 인덱스에 맞춰 반환
    result = pd.Series(index=data.index, dtype='category')
    result.loc[returns.index] = labels
    
    return result


def create_regression_labels(
    data: pd.DataFrame,
    return_col: str,
    normalize: bool = True,
    winsorize: Optional[Tuple[float, float]] = (0.05, 0.95)
) -> pd.Series:
    """회귀 레이블을 생성합니다.
    
    Args:
        data: 데이터프레임
        return_col: 수익률 컬럼명
        normalize: 정규화 여부 (Z-스코어)
        winsorize: 이상치 처리 분위수 (하한, 상한)
        
    Returns:
        회귀 레이블 시리즈
    """
    labels = data[return_col].copy()
    
    # 이상치 처리 (Winsorization)
    if winsorize is not None:
        lower_bound = labels.quantile(winsorize[0])
        upper_bound = labels.quantile(winsorize[1])
        labels = labels.clip(lower_bound, upper_bound)
    
    # 정규화
    if normalize:
        labels = (labels - labels.mean()) / labels.std()
    
    return labels


def create_ranking_labels(
    data: pd.DataFrame,
    return_col: str,
    method: str = 'dense'
) -> pd.Series:
    """랭킹 레이블을 생성합니다 (포트폴리오 최적화용).
    
    Args:
        data: 데이터프레임 (cross-sectional 데이터)
        return_col: 수익률 컬럼명
        method: 랭킹 방법 ('dense', 'min', 'max', 'average', 'first')
        
    Returns:
        랭킹 레이블 시리즈
    """
    return data[return_col].rank(method=method, ascending=False)


def create_binary_labels(
    data: pd.DataFrame,
    return_col: str,
    threshold: float = 0.0,
    positive_class: str = 'up',
    negative_class: str = 'down'
) -> pd.Series:
    """이진 분류 레이블을 생성합니다.
    
    Args:
        data: 데이터프레임
        return_col: 수익률 컬럼명
        threshold: 분류 임계값
        positive_class: 양의 클래스 레이블
        negative_class: 음의 클래스 레이블
        
    Returns:
        이진 레이블 시리즈
    """
    returns = data[return_col]
    
    labels = pd.Series(index=data.index, dtype='category')
    labels[returns > threshold] = positive_class
    labels[returns <= threshold] = negative_class
    
    return labels


def add_all_labels(
    data: pd.DataFrame,
    periods: List[int] = [1, 5, 10, 20],
    classification_methods: Dict[str, Dict] = None,
    price_col: str = 'close'
) -> pd.DataFrame:
    """모든 종류의 레이블을 추가합니다.
    
    Args:
        data: 입력 데이터프레임
        periods: 예측 기간 리스트
        classification_methods: 분류 방법 설정
        price_col: 가격 컬럼명
        
    Returns:
        레이블이 추가된 데이터프레임
    """
    logger.info(f"🏷️ 레이블 생성 시작: {len(data)} 행")
    
    if classification_methods is None:
        classification_methods = {
            'quantile_3': {'method': 'quantile', 'n_classes': 3},
            'quantile_5': {'method': 'quantile', 'n_classes': 5},
            'threshold': {'method': 'threshold', 'thresholds': [-0.02, 0.02]},
            'binary': {'method': 'threshold', 'thresholds': [0.0]}
        }
    
    result = data.copy()
    
    # 미래 수익률 계산
    result = calculate_forward_returns(result, periods, price_col)
    
    # 각 기간별로 다양한 레이블 생성
    for period in periods:
        return_col = f'forward_return_{period}d'
        
        if return_col not in result.columns:
            continue
        
        # 회귀 레이블
        result[f'label_reg_{period}d'] = create_regression_labels(
            result, return_col, normalize=True
        )
        
        # 분류 레이블들
        for label_name, config in classification_methods.items():
            method = config['method']
            
            if method == 'quantile':
                labels = create_classification_labels(
                    result, return_col, method, config['n_classes']
                )
                result[f'label_{label_name}_{period}d'] = labels
                
            elif method == 'threshold':
                thresholds = config['thresholds']
                if len(thresholds) == 1:  # 이진 분류
                    labels = create_binary_labels(
                        result, return_col, thresholds[0]
                    )
                    result[f'label_binary_{period}d'] = labels
                else:  # 다중 분류
                    labels = create_classification_labels(
                        result, return_col, method, thresholds=thresholds
                    )
                    result[f'label_{label_name}_{period}d'] = labels
    
    logger.success(f"✅ 레이블 생성 완료: {len([c for c in result.columns if c.startswith('label_')])} 개")
    return result


def add_cross_sectional_labels(
    data: pd.DataFrame,
    date_col: str = 'date',
    return_cols: List[str] = None,
    ranking_percentiles: List[int] = [10, 20, 80, 90]
) -> pd.DataFrame:
    """횡단면(Cross-sectional) 레이블을 추가합니다.
    
    각 날짜별로 종목들을 수익률로 랭킹하여 상대적 성과 레이블을 생성합니다.
    
    Args:
        data: MultiIndex (date, ticker) 데이터프레임
        date_col: 날짜 컬럼명 (인덱스가 아닌 경우)
        return_cols: 수익률 컬럼 리스트
        ranking_percentiles: 랭킹 분위수 리스트
        
    Returns:
        횡단면 레이블이 추가된 데이터프레임
    """
    logger.info("🔄 횡단면 레이블 생성 시작")
    
    if return_cols is None:
        return_cols = [col for col in data.columns if col.startswith('forward_return_')]
    
    result = data.copy()
    
    # 인덱스가 MultiIndex인지 확인
    if isinstance(data.index, pd.MultiIndex):
        # MultiIndex 데이터 처리
        for return_col in return_cols:
            period = return_col.split('_')[-1]  # 기간 추출
            
            # 각 날짜별 랭킹
            rankings = data.groupby(level=0)[return_col].rank(
                method='dense', ascending=False, pct=True
            )
            
            # 분위수 기반 분류
            for percentile in ranking_percentiles:
                threshold = percentile / 100.0
                if percentile <= 50:
                    # 하위 분위수 (예: 10% = 상위 10%)
                    label_name = f'top_{percentile}pct_{period}'
                    result[label_name] = (rankings <= threshold).astype(int)
                else:
                    # 상위 분위수 (예: 90% = 하위 10%)
                    label_name = f'bottom_{100-percentile}pct_{period}'
                    result[label_name] = (rankings >= threshold).astype(int)
    
    else:
        # 일반 데이터프레임인 경우
        if date_col not in data.columns:
            logger.warning("날짜 컬럼을 찾을 수 없습니다. 횡단면 레이블을 건너뜁니다.")
            return result
        
        for return_col in return_cols:
            period = return_col.split('_')[-1]
            
            # 각 날짜별 랭킹
            rankings = data.groupby(date_col)[return_col].rank(
                method='dense', ascending=False, pct=True
            )
            
            # 분위수 기반 분류
            for percentile in ranking_percentiles:
                threshold = percentile / 100.0
                if percentile <= 50:
                    label_name = f'top_{percentile}pct_{period}'
                    result[label_name] = (rankings <= threshold).astype(int)
                else:
                    label_name = f'bottom_{100-percentile}pct_{period}'
                    result[label_name] = (rankings >= threshold).astype(int)
    
    logger.success("✅ 횡단면 레이블 생성 완료")
    return result


def validate_labels(data: pd.DataFrame, label_cols: List[str] = None) -> Dict[str, Any]:
    """레이블 품질을 검증합니다.
    
    Args:
        data: 데이터프레임
        label_cols: 검증할 레이블 컬럼 리스트
        
    Returns:
        검증 결과 딕셔너리
    """
    if label_cols is None:
        label_cols = [col for col in data.columns if col.startswith('label_')]
    
    validation_results = {}
    
    for col in label_cols:
        if col not in data.columns:
            continue
        
        col_data = data[col].dropna()
        
        result = {
            'total_count': len(data),
            'valid_count': len(col_data),
            'missing_count': len(data) - len(col_data),
            'missing_rate': (len(data) - len(col_data)) / len(data),
        }
        
        if len(col_data) > 0:
            if col_data.dtype in ['object', 'category']:
                # 분류 레이블
                result['unique_values'] = col_data.nunique()
                result['value_counts'] = col_data.value_counts().to_dict()
                result['class_balance'] = (col_data.value_counts() / len(col_data)).to_dict()
            else:
                # 회귀 레이블
                result['mean'] = col_data.mean()
                result['std'] = col_data.std()
                result['min'] = col_data.min()
                result['max'] = col_data.max()
                result['skewness'] = col_data.skew()
                result['kurtosis'] = col_data.kurtosis()
        
        validation_results[col] = result
    
    return validation_results


def print_label_summary(validation_results: Dict[str, Any]):
    """레이블 검증 결과를 출력합니다.
    
    Args:
        validation_results: validate_labels 함수의 결과
    """
    logger.info("📋 레이블 검증 결과:")
    
    for label_name, results in validation_results.items():
        logger.info(f"\n🏷️ {label_name}:")
        logger.info(f"  총 개수: {results['total_count']:,}")
        logger.info(f"  유효 개수: {results['valid_count']:,}")
        logger.info(f"  결측률: {results['missing_rate']:.2%}")
        
        if 'unique_values' in results:
            # 분류 레이블
            logger.info(f"  클래스 수: {results['unique_values']}")
            logger.info("  클래스 분포:")
            for value, count in results['value_counts'].items():
                percentage = results['class_balance'][value]
                logger.info(f"    {value}: {count:,} ({percentage:.2%})")
        else:
            # 회귀 레이블
            logger.info(f"  평균: {results['mean']:.4f}")
            logger.info(f"  표준편차: {results['std']:.4f}")
            logger.info(f"  범위: [{results['min']:.4f}, {results['max']:.4f}]")


# CLI 직접 실행용
if __name__ == "__main__":
    # 테스트용 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100)
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    test_data = []
    for ticker in tickers:
        for date in dates:
            price = 100 + np.random.randn() * 10
            test_data.append({
                'date': date,
                'ticker': ticker,
                'close': price
            })
    
    df = pd.DataFrame(test_data)
    df = df.set_index(['date', 'ticker'])
    
    # 레이블 생성 테스트
    labeled_data = add_all_labels(df)
    
    # 검증
    validation = validate_labels(labeled_data)
    print_label_summary(validation)
    
    print(f"\n✅ 테스트 완료: {len(labeled_data.columns)} 컬럼 생성")