"""
통계 피처 추출 모듈

이 모듈은 다음을 담당합니다:
  - 슬라이딩 윈도우 기반 통계량 추출 (단일 윈도우 크기)
  - 멀티 스케일 통계 피처 추출 (여러 윈도우 크기 병합)
  - Strategy A : 연속형 6통계(mean/std/min/max/median/range)
                 + 이산형 1통계(ratio) → 45 컬럼
  - Strategy B : PCA 주성분 7통계(mean/std/min/max/range/skew/kurt) 추출
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

    Notes
    -----
    'range'는 pandas rolling이 직접 지원하지 않으므로 이 함수에서는
    처리되지 않습니다. range는 호출부에서 max - min으로 직접 계산합니다.
    """
    roller = df[cols].rolling(window=window_size, min_periods=1)
    parts = []
    for stat in stats:
        agg: pd.DataFrame = getattr(roller, stat)()
        agg.columns = [f'{col}_{stat}' for col in cols]
        parts.append(agg)
    return parts


def _rolling_range(
    df: pd.DataFrame,
    cols: list[str],
    window_size: int,
) -> pd.DataFrame:
    """
    롤링 range(= max - min)를 계산합니다.

    pandas rolling이 range를 직접 지원하지 않아 max와 min을 각각 계산한 뒤
    차이를 구합니다. max/min rolling 객체를 재사용해 중복 연산을 줄입니다.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str]
        range를 계산할 컬럼 목록
    window_size : int

    Returns
    -------
    pd.DataFrame
        컬럼명 형식: '{col}_range'
    """
    roller = df[cols].rolling(window=window_size, min_periods=1)
    range_df = roller.max() - roller.min()
    range_df.columns = [f'{col}_range' for col in cols]
    return range_df


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
        'A': 연속형(mean/std/min/max/median/range) + 이산형(ratio)
        'B': PCA 주성분(mean/std/min/max/range/skew/kurt)

    Returns
    -------
    pd.DataFrame
        컬럼명 형식: '{채널명}_{통계량종류}'
          예) x_3a_mean, x_7e_std, x_06_ratio, pc_01_skew
        strategy='A' : (7채널 × 6통계) + (3채널 × 1통계) = 45 컬럼
        strategy='B' : (PCA 주성분 수 × 7통계) 컬럼

    통계량별 탐지 목표
    ------------------
    mean   : level shift (평균 이동)
    std    : 진폭 변화
    min/max: spike, dip
    median : spike에 robust한 level (mean과 함께 쓰면 spike 여부 판별 가능)
    range  : 구간 내 진폭 — spike/dip 탐지에 직접 유리
    skew   : 분포 비대칭성 — Collective anomaly 시 분포 대칭성 붕괴 포착 (B 전용)
    kurt   : 분포 꼬리 두께 — 이상값 유입 시 정규분포보다 꼬리 두꺼워짐 (B 전용)

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

    # ── Strategy A ───────────────────────────────────────────────────────
    if strategy == 'A':
        feature_cols = [c for c in df.columns if c.startswith('x_')]
        if not feature_cols:
            raise ValueError("strategy='A'를 위한 'x_'로 시작하는 컬럼이 없습니다.")

        continuous_cols = [c for c in feature_cols if c not in DISCRETE_COLS]
        discrete_cols   = [c for c in feature_cols if c in DISCRETE_COLS]

        # 연속형 채널: 5개 기본 통계량 → 7채널 × 5 = 35 컬럼
        feature_parts.extend(
            _rolling_stats(df, continuous_cols, window_size,
                           ['mean', 'std', 'min', 'max', 'median'])
        )

        # range = max - min (구간 내 진폭) → 7채널 × 1 = 7 컬럼
        # spike/dip 탐지에 직접 유리 (수업 자료 권장 feature)
        feature_parts.append(
            _rolling_range(df, continuous_cols, window_size)
        )

        if discrete_cols:
            # 이산형 채널: mean만 추출 (구간 내 활성화 비율) → 3채널 × 1 = 3 컬럼
            # std/min/max는 이진 신호에서 고정값에 가까워 노이즈이므로 제외
            ratio = df[discrete_cols].rolling(window=window_size, min_periods=1).mean()
            ratio.columns = [f'{col}_ratio' for col in discrete_cols]
            feature_parts.append(ratio)

    # ── Strategy B ───────────────────────────────────────────────────────
    else:
        pca_cols = [c for c in df.columns if c.startswith('pc_')]
        if not pca_cols:
            raise ValueError("strategy='B'를 위한 'pc_'로 시작하는 PCA 컬럼이 없습니다.")

        roller = df[pca_cols].rolling(window=window_size, min_periods=1)

        # 기본 4개 통계량
        feature_parts.extend(
            _rolling_stats(df, pca_cols, window_size, ['mean', 'std', 'min', 'max'])
        )

        # range: 진폭 이상 포착 — Collective anomaly에서 진폭이 비정상적으로 커질 때 유리
        feature_parts.append(
            _rolling_range(df, pca_cols, window_size)
        )

        # 분포 형태 통계량: 왜도(skew), 첨도(kurt)
        # 밀도 기반 이상탐지(OCSVM·GMM)가 분포의 비대칭성과 꼬리 두께를 학습하도록 추가
        # 작은 window에서 NaN 발생 → 정상 분포 기준값(0)으로 채움
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
        적용할 윈도우 크기 목록 (예: [50, 200, 500])
    strategy : str, default 'A'
        extract_statistical_features()에 전달할 strategy

    Returns
    -------
    pd.DataFrame
        컬럼명 형식: '{채널명}_{통계량종류}_w{윈도우크기}'
          예) x_3a_mean_w50, x_06_ratio_w200, pc_01_skew_w500

    Notes
    -----
    실험 계획상 window 크기별로 별도 모델을 학습하는 경우
    (IF: w=50 / OCSVM·GMM: w=200, 500) 이 함수보다
    extract_statistical_features()를 window_size별로 직접 호출하는 것이 적합합니다.
    이 함수는 멀티스케일 피처를 하나의 모델에 동시에 입력하거나
    앙상블 실험(Exp 6)에서 활용합니다.

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