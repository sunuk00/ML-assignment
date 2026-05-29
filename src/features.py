"""
통계 피처 추출 모듈

이 모듈은 다음을 담당합니다:
  - 슬라이딩 윈도우 기반 통계량 추출 (단일 윈도우 크기)
  - 멀티 스케일 통계 피처 추출 (여러 윈도우 크기 병합)
  - Strategy A : 연속형 4통계(mean/std/min/max) + 이산형 1통계(ratio) → 31 컬럼
  - Strategy B : PCA 주성분 6통계(mean/std/min/max/skew/kurt) 추출
"""

from __future__ import annotations

import pandas as pd


# 이산형(Discrete) 채널 이름 목록
# preprocessing.py의 DISCRETE_COLS와 항상 동일하게 유지해야 합니다
DISCRETE_COLS: list[str] = ['x_06', 'x_92', 'x_4b']


def _rolling_stats(
    df: pd.DataFrame,
    cols: list[str],
    window_size: int,
    stats: list[str],
) -> list[pd.DataFrame]:
    """
    지정된 컬럼에 대해 여러 롤링 통계량을 계산하여 DataFrame 리스트로 반환합니다.

    컬럼명은 '{원래컬럼명}_{통계량종류}' 형태로 자동 변환됩니다.
    """
    roller = df[cols].rolling(window=window_size, min_periods=1)
    parts = []
    for stat in stats:
        agg: pd.DataFrame = getattr(roller, stat)()
        agg.columns = [f'{col}_{stat}' for col in cols]
        parts.append(agg)
    return parts


def extract_statistical_features(
    df: pd.DataFrame,
    window_size: int,
    strategy: str = 'A',
) -> pd.DataFrame:
    """
    슬라이딩 윈도우 기반 통계 피처를 추출합니다.

    Parameters
    ----------
    df : pd.DataFrame
        process_pipeline()의 출력 데이터프레임
          - strategy='A' : x_로 시작하는 feature 컬럼 10개
          - strategy='B' : pc_로 시작하는 PCA 주성분 컬럼
    window_size : int
        슬라이딩 윈도우 크기 (min_periods=1로 시작 구간도 계산)
    strategy : str, default 'A'
        'A': 연속형(mean/std/min/max) + 이산형(ratio)
        'B': PCA 주성분(mean/std/min/max/skew/kurt)

    Returns
    -------
    pd.DataFrame
        컬럼명 형식: '{채널명}_{통계량종류}'
          예) x_3a_mean, x_7e_std, x_06_ratio, pc_01_skew
        strategy='A' : (7채널 × 4통계) + (3채널 × 1통계) = 31 컬럼
        strategy='B' : (PCA 주성분 수 × 6통계) 컬럼

    Raises
    ------
    ValueError
        window_size가 1 미만이거나 strategy가 유효하지 않은 경우
        해당 strategy에 맞는 컬럼이 df에 없는 경우
    """
    if window_size < 1:
        raise ValueError(
            f"window_size는 1 이상이어야 합니다. 입력값: {window_size}"
        )
    if strategy not in ('A', 'B'):
        raise ValueError(
            f"strategy는 'A' 또는 'B'여야 합니다. 입력값: {strategy!r}"
        )

    feature_parts: list[pd.DataFrame] = []

    if strategy == 'A':
        feature_cols = [c for c in df.columns if c.startswith('x_')]
        if not feature_cols:
            raise ValueError("strategy='A'를 위한 'x_'로 시작하는 컬럼이 없습니다.")

        continuous_cols = [c for c in feature_cols if c not in DISCRETE_COLS]
        discrete_cols   = [c for c in feature_cols if c in DISCRETE_COLS]

        # 연속형 채널: 4개 통계량 → 7채널 × 4 = 28 컬럼
        feature_parts.extend(
            _rolling_stats(df, continuous_cols, window_size, ['mean', 'std', 'min', 'max'])
        )

        if discrete_cols:
            # 이산형 채널: mean만 추출 (구간 내 활성화 비율) → 3채널 × 1 = 3 컬럼
            # std/min/max는 이진 신호에서 노이즈이므로 제외
            ratio = df[discrete_cols].rolling(window=window_size, min_periods=1).mean()
            ratio.columns = [f'{col}_ratio' for col in discrete_cols]
            feature_parts.append(ratio)

    else:  # strategy == 'B'
        pca_cols = [c for c in df.columns if c.startswith('pc_')]
        if not pca_cols:
            raise ValueError("strategy='B'를 위한 'pc_'로 시작하는 PCA 컬럼이 없습니다.")

        roller = df[pca_cols].rolling(window=window_size, min_periods=1)

        # 기본 4개 통계량
        feature_parts.extend(
            _rolling_stats(df, pca_cols, window_size, ['mean', 'std', 'min', 'max'])
        )

        # 분포 형태 통계량: 왜도(skew), 첨도(kurt)
        # 밀도 기반 이상탐지 모델이 분포의 비대칭성과 꼬리 두께를 학습하도록 추가
        skew_df = roller.skew().fillna(0)
        skew_df.columns = [f'{col}_skew' for col in pca_cols]
        feature_parts.append(skew_df)

        kurt_df = roller.kurt().fillna(0)
        kurt_df.columns = [f'{col}_kurt' for col in pca_cols]
        feature_parts.append(kurt_df)

    result = pd.concat(feature_parts, axis=1)
    # std는 단일 관측치 구간에서 NaN 발생 → 0으로 채움
    return result.fillna(0)


def multi_scale_extract(
    df: pd.DataFrame,
    window_sizes: list[int],
    strategy: str = 'A',
) -> pd.DataFrame:
    """
    여러 윈도우 크기에 대해 통계 피처를 추출하고 수평으로 병합합니다.

    단기 패턴(작은 윈도우)과 장기 패턴(큰 윈도우)을 동시에 포착하여
    이상징후 탐지 성능을 높이는 멀티 스케일 접근법입니다.

    Parameters
    ----------
    df : pd.DataFrame
        process_pipeline()의 출력 데이터프레임
    window_sizes : list[int]
        적용할 윈도우 크기 목록 (예: [10, 30, 50])
    strategy : str, default 'A'
        extract_statistical_features()에 전달할 strategy

    Returns
    -------
    pd.DataFrame
        컬럼명 형식: '{채널명}_{통계량종류}_w{윈도우크기}'
          예) x_3a_mean_w10, x_06_ratio_w30, pc_01_skew_w50

    Raises
    ------
    ValueError
        window_sizes가 비어 있는 경우
    """
    if not window_sizes:
        raise ValueError("window_sizes는 비어 있으면 안 됩니다.")

    parts = []
    for ws in window_sizes:
        feat = extract_statistical_features(df, ws, strategy)
        feat.columns = [f'{col}_w{ws}' for col in feat.columns]
        parts.append(feat)

    return pd.concat(parts, axis=1)
