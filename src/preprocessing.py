from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# EDA에서 확인된 이산형(바이너리) 채널 3개
DISCRETE_COLS: list[str] = ['x_06', 'x_92', 'x_4b']


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치를 선형 보간으로 처리합니다.

    보간 후 앞/뒤에 남는 결측치는 bfill → ffill 순서로 제거합니다.
    결측치가 없으면 복사본만 반환합니다.

    Examples
    --------
    >>> train_df = fill_missing(train_df)
    >>> val_df   = fill_missing(val_df)
    """
    if df.isnull().sum().sum() == 0:
        return df.copy()
    return df.copy().interpolate(method='linear').bfill().ffill()


def filter_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """
    이산형 채널(DISCRETE_COLS)을 제거하고 연속형 채널만 반환합니다.

    Strategy B 전처리의 첫 번째 단계입니다.
    Strategy A는 이 함수를 건너뛰고 전체 채널을 그대로 사용합니다.

    Examples
    --------
    >>> train_cont = filter_continuous(train_df)   # x_06, x_92, x_4b 제거
    """
    feature_cols = [c for c in df.columns if c.startswith('x_')]
    continuous_cols = [c for c in feature_cols if c not in DISCRETE_COLS]
    return df[continuous_cols].copy()


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """
    학습 데이터로 StandardScaler를 fit하고 반환합니다.

    반드시 train 데이터로만 fit하세요. Val/Test에는 apply_scaler를 사용합니다.
    (train으로 fit한 scaler를 val/test에도 동일하게 적용 → Data Leakage 방지)

    Examples
    --------
    >>> scaler = fit_scaler(train_cont)
    """
    scaler = StandardScaler()
    scaler.fit(df.values)
    return scaler


def apply_scaler(scaler: StandardScaler, df: pd.DataFrame) -> pd.DataFrame:
    """
    fit된 StandardScaler로 데이터를 transform합니다.

    train, val, test 모두 동일한 scaler 객체를 사용해야 합니다.

    Examples
    --------
    >>> train_scaled = apply_scaler(scaler, train_cont)
    >>> val_scaled   = apply_scaler(scaler, val_cont)   # 같은 scaler!
    """
    scaled = scaler.transform(df.values)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def fit_pca(
    df: pd.DataFrame,
    variance: float = 0.95,
    random_state: int = 42,
) -> PCA:
    """
    학습 데이터로 PCA를 fit하고 반환합니다.

    Parameters
    ----------
    df       : apply_scaler()의 출력 DataFrame
    variance : 유지할 누적 분산 비율 (기본 95% → 주성분 수 자동 결정)

    반드시 train 데이터로만 fit하세요.

    Examples
    --------
    >>> pca = fit_pca(train_scaled)
    >>> print(f"주성분 수: {pca.n_components_}")
    """
    pca = PCA(n_components=variance, random_state=random_state)
    pca.fit(df.values)
    return pca


def apply_pca(pca: PCA, df: pd.DataFrame) -> pd.DataFrame:
    """
    fit된 PCA로 데이터를 transform합니다.

    컬럼명은 pc_01, pc_02, ... 형식으로 자동 부여됩니다.
    train, val, test 모두 동일한 pca 객체를 사용해야 합니다.

    Examples
    --------
    >>> train_pca = apply_pca(pca, train_scaled)
    >>> val_pca   = apply_pca(pca, val_scaled)    # 같은 pca!
    """
    transformed = pca.transform(df.values)
    n = transformed.shape[1]
    cols = [f'pc_{i + 1:02d}' for i in range(n)]
    return pd.DataFrame(transformed, index=df.index, columns=cols)
