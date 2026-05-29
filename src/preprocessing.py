"""
전처리 파이프라인 모듈

이 모듈은 다음을 담당합니다:
  - 결측치 처리 (선형 보간 + bfill/ffill)
  - Strategy A : 10개 채널 전체 유지 (스케일링/PCA 미적용)
  - Strategy B : 연속형 7채널 필터링 → StandardScaler → PCA(95% 누적 분산) 적용
  - Data Leakage 방지 : Train은 fit_transform, Val/Test는 transform만 수행
"""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# 이산형(Discrete) 채널 이름 목록
# 실제 데이터에서 확인된 이진(binary) 채널 3개
# 데이터셋이 변경될 경우 이 리스트만 수정하면 됩니다
DISCRETE_COLS: list[str] = ['x_06', 'x_92', 'x_4b']


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    시계열 결측치를 선형 보간법으로 처리합니다.

    선형 보간 후 앞/뒤에 남는 결측치는 bfill → ffill 순서로 제거합니다.
    시계열의 시간적 의존성을 보존하기 위해 linear 방식을 사용합니다.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임

    Returns
    -------
    pd.DataFrame
        결측치가 없는 데이터프레임 (원본 인덱스 유지)
    """
    if df.isnull().sum().sum() == 0:
        return df.copy()

    result = df.copy()
    result = result.interpolate(method='linear')
    result = result.bfill().ffill()
    return result


def process_pipeline(
    df: pd.DataFrame,
    strategy: str = 'A',
    scaler: StandardScaler | None = None,
    pca: PCA | None = None,
) -> tuple[pd.DataFrame, StandardScaler | None, PCA | None]:
    """
    전처리 파이프라인을 실행합니다.

    Parameters
    ----------
    df : pd.DataFrame
        feature 컬럼만 포함된 데이터프레임 (t, label 컬럼 제외 권장)
    strategy : str, default 'A'
        'A': 10개 채널 전체 유지, 스케일링/PCA 미적용
        'B': 연속형 7채널 필터링 → StandardScaler → PCA(95%) 적용
    scaler : StandardScaler or None
        None이면 Train 모드(fit_transform), 전달되면 Val/Test 모드(transform only)
    pca : PCA or None
        None이면 Train 모드(fit_transform), 전달되면 Val/Test 모드(transform only)

    Returns
    -------
    processed_df : pd.DataFrame
        변환된 데이터프레임
    scaler : StandardScaler or None
        Strategy B에서 학습된 스케일러 객체 (A면 None)
    pca : PCA or None
        Strategy B에서 학습된 PCA 객체 (A면 None)

    Raises
    ------
    ValueError
        strategy가 'A' 또는 'B'가 아닌 경우
        Val/Test 모드에서 scaler와 pca 중 하나만 None인 경우
    KeyError
        DISCRETE_COLS에 지정된 컬럼이 df에 없는 경우
    """
    if strategy not in ('A', 'B'):
        raise ValueError(
            f"strategy는 'A' 또는 'B'여야 합니다. 입력값: {strategy!r}"
        )

    feature_cols = [c for c in df.columns if c.startswith('x_')]
    if not feature_cols:
        raise ValueError("'x_'로 시작하는 feature 컬럼이 없습니다.")

    # ---------- Strategy A: 전체 채널 그대로 반환 ----------
    if strategy == 'A':
        return df[feature_cols].copy(), None, None

    # ---------- Strategy B: 연속형 채널 → StandardScaler → PCA ----------

    missing_cols = [c for c in DISCRETE_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"DISCRETE_COLS에 지정된 컬럼이 데이터프레임에 없습니다: {missing_cols}"
        )

    # scaler와 pca의 None 여부가 불일치하면 사용자 실수 가능성이 높으므로 오류 처리
    if (scaler is None) != (pca is None):
        raise ValueError(
            "scaler와 pca는 둘 다 None(Train 모드)이거나 "
            "둘 다 객체(Val/Test 모드)여야 합니다."
        )

    continuous_cols = [c for c in feature_cols if c not in DISCRETE_COLS]
    X = df[continuous_cols].values  # shape: (T, 7)

    is_train = scaler is None

    if is_train:
        # Train: fit_transform — scaler/pca를 새로 학습
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
    else:
        # Val/Test: transform only — Data Leakage 방지
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

    n_components = X_pca.shape[1]
    pca_cols = [f'pc_{i + 1:02d}' for i in range(n_components)]
    processed_df = pd.DataFrame(X_pca, index=df.index, columns=pca_cols)

    return processed_df, scaler, pca
