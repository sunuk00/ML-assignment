"""
통계 피처 추출 모듈

실험 스크립트에서 직접 조합해서 쓸 수 있도록 낮은 레벨로 설계되어 있습니다.

공개 API
--------
rolling_features(df, cols, window_size, stats)
    pandas rolling 기반. timestep 길이 보존.
window_features(df, cols, window_size, stride, stats)
    sliding window 기반. timestep 길이 보존 (mean 방식으로 환산).
make_windows(values, window_size, stride)
    시계열 → sliding window 배열 변환.

사용 가능한 통계량
-----------------
'mean', 'std', 'min', 'max', 'median', 'range', 'skew', 'kurt'

이산형 채널의 ratio(1의 비율)는 stats=['mean']으로 계산합니다.

예시
----
>>> from src.features import rolling_features, window_features, DISCRETE_COLS
>>>
>>> x_cols   = [c for c in train_df.columns if c.startswith('x_')]
>>> cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
>>> disc_cols = [c for c in x_cols if c in DISCRETE_COLS]
>>>
>>> # 연속형: 6통계, 이산형: ratio(mean)
>>> feats = pd.concat([
...     rolling_features(train_df, cols=cont_cols, window_size=50,
...                      stats=['mean', 'std', 'min', 'max', 'median', 'range']),
...     rolling_features(train_df, cols=disc_cols, window_size=50,
...                      stats=['mean']),
... ], axis=1)
>>>
>>> # PCA 성분에 skew/kurt 추가
>>> pc_cols = [c for c in train_pca.columns if c.startswith('pc_')]
>>> feats = rolling_features(train_pca, cols=pc_cols, window_size=200,
...                          stats=['mean', 'std', 'min', 'max', 'range', 'skew', 'kurt'])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt

# EDA에서 확인된 이산형(바이너리) 채널
DISCRETE_COLS: list[str] = ['x_06', 'x_92', 'x_4b']


# ──────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────

def rolling_features(
    df: pd.DataFrame,
    cols: list[str],
    window_size: int,
    stats: list[str] = ('mean', 'std', 'min', 'max', 'range'),
) -> pd.DataFrame:
    """
    지정한 컬럼에 대해 pandas rolling 통계량을 계산합니다.

    Parameters
    ----------
    df          : 입력 DataFrame
    cols        : 통계량을 계산할 컬럼 이름 리스트
    window_size : 윈도우 크기 (min_periods=1로 앞쪽 패딩)
    stats       : 계산할 통계량. 지원 목록:
                  'mean', 'std', 'min', 'max', 'median', 'range', 'skew', 'kurt'

    Returns
    -------
    pd.DataFrame  컬럼명 형식: '{col}_{stat}', 행 수 = 입력과 동일

    Examples
    --------
    >>> feats = rolling_features(train_df, cols=cont_cols, window_size=50,
    ...                          stats=['mean', 'std', 'range'])
    """
    parts: dict[str, np.ndarray] = {}
    for col in cols:
        for stat, arr in _rolling_one_col(df[col].to_numpy(), window_size, list(stats)).items():
            parts[f'{col}_{stat}'] = arr
    return pd.DataFrame(parts, index=df.index).fillna(0)


def window_features(
    df: pd.DataFrame,
    cols: list[str],
    window_size: int,
    stride: int = 1,
    stats: list[str] = ('mean', 'std', 'min', 'max', 'range'),
) -> pd.DataFrame:
    """
    지정한 컬럼에 대해 sliding window 통계량을 계산합니다.

    각 window의 통계량을 해당 window가 포함하는 모든 timestep에 누적 평균으로 환산하므로
    반환 DataFrame의 행 수는 입력과 동일합니다.

    Parameters
    ----------
    df          : 입력 DataFrame
    cols        : 통계량을 계산할 컬럼 이름 리스트
    window_size : 윈도우 크기
    stride      : 슬라이딩 보폭
    stats       : 계산할 통계량. 지원 목록:
                  'mean', 'std', 'min', 'max', 'median', 'range', 'skew', 'kurt'

    Returns
    -------
    pd.DataFrame  컬럼명 형식: '{col}_{stat}', 행 수 = 입력과 동일

    Examples
    --------
    >>> feats = window_features(train_df, cols=cont_cols, window_size=100, stride=10,
    ...                         stats=['mean', 'std', 'skew', 'kurt'])
    """
    parts: dict[str, np.ndarray] = {}
    for col in cols:
        for stat, arr in _window_one_col(df[col].to_numpy(), window_size, stride, list(stats)).items():
            parts[f'{col}_{stat}'] = arr
    return pd.DataFrame(parts, index=df.index).fillna(0)


def make_windows(
    values: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """
    시계열을 sliding window 배열로 변환합니다.

    Returns
    -------
    (T,)   → (N, window_size)
    (T, D) → (N, window_size, D)
    N = (T - window_size) // stride + 1

    Examples
    --------
    >>> windows = make_windows(series, window_size=50, stride=1)  # (N, 50)
    """
    values = np.asarray(values)
    T = values.shape[0]
    if T < window_size:
        raise ValueError(f"입력 길이({T})가 window_size({window_size})보다 짧습니다.")
    n = (T - window_size) // stride + 1
    if values.ndim == 1:
        return np.stack([values[i*stride : i*stride+window_size] for i in range(n)])
    elif values.ndim == 2:
        return np.stack([values[i*stride : i*stride+window_size, :] for i in range(n)])
    raise ValueError(f"지원하지 않는 차원: {values.ndim}")


# ──────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────

def _rolling_one_col(
    series: np.ndarray,
    window_size: int,
    stats: list[str],
) -> dict[str, np.ndarray]:
    s = pd.Series(series)
    r = s.rolling(window=window_size, min_periods=1)
    result: dict[str, np.ndarray] = {}
    _max = _min = None
    for stat in stats:
        if stat == 'range':
            if _max is None: _max = r.max().to_numpy()
            if _min is None: _min = r.min().to_numpy()
            result['range'] = _max - _min
        elif stat == 'skew':
            result['skew'] = r.skew().fillna(0).to_numpy()
        elif stat == 'kurt':
            result['kurt'] = r.kurt().fillna(0).to_numpy()
        else:
            result[stat] = getattr(r, stat)().to_numpy()
    return result


def _window_one_col(
    series: np.ndarray,
    window_size: int,
    stride: int,
    stats: list[str],
) -> dict[str, np.ndarray]:
    T = len(series)
    windows = make_windows(series, window_size, stride)
    result: dict[str, np.ndarray] = {}
    _max_w = _min_w = None
    for stat in stats:
        if stat == 'mean':
            per_window = windows.mean(axis=1)
        elif stat == 'std':
            per_window = windows.std(axis=1)
        elif stat == 'min':
            _min_w = windows.min(axis=1)
            per_window = _min_w
        elif stat == 'max':
            _max_w = windows.max(axis=1)
            per_window = _max_w
        elif stat == 'median':
            per_window = np.median(windows, axis=1)
        elif stat == 'range':
            if _max_w is None: _max_w = windows.max(axis=1)
            if _min_w is None: _min_w = windows.min(axis=1)
            per_window = _max_w - _min_w
        elif stat == 'skew':
            per_window = np.apply_along_axis(
                lambda x: float(scipy_skew(x)) if x.std() > 1e-8 else 0.0,
                1, windows,
            )
        elif stat == 'kurt':
            per_window = np.apply_along_axis(
                lambda x: float(scipy_kurt(x)) if x.std() > 1e-8 else 0.0,
                1, windows,
            )
        else:
            raise ValueError(f"알 수 없는 통계량: {stat}")
        result[stat] = _assign_window_scores(per_window, T, window_size, stride)
    return result


def _assign_window_scores(
    window_scores: np.ndarray,
    T: int,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """각 timestep을 포함하는 모든 window score의 평균으로 timestep score 환산."""
    score_sum = np.zeros(T, dtype=float)
    count     = np.zeros(T, dtype=float)
    for i, sc in enumerate(window_scores):
        s = i * stride
        e = min(s + window_size, T)
        score_sum[s:e] += sc
        count[s:e]     += 1.0
    with np.errstate(invalid='ignore', divide='ignore'):
        scores = np.where(count > 0, score_sum / count, np.nan)
    for i in range(1, T):
        if np.isnan(scores[i]): scores[i] = scores[i-1]
    for i in range(T-2, -1, -1):
        if np.isnan(scores[i]): scores[i] = scores[i+1]
    return scores
