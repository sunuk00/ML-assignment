from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def flip_score(scores: np.ndarray) -> np.ndarray:
    """
    '작을수록 이상'인 score를 '클수록 이상'으로 반전합니다.

    IF(score_samples), OCSVM(decision_function), GMM(score_samples)
    세 모델 모두 원본 score가 작을수록 이상이므로 이 함수를 공통으로 사용합니다.

    Examples
    --------
    >>> raw    = model.score_samples(X_val)  # 작을수록 이상
    >>> scores = flip_score(raw)             # 클수록 이상
    """
    return -1.0 * np.asarray(scores)


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Score 배열을 rank 기반으로 [0, 1]로 정규화합니다.

    모델마다 score의 분포와 scale이 달라 직접 평균이 불가능합니다.
    rank로 변환하면 scale이 통일되어 앙상블이 가능해집니다.
    동점은 평균 rank로 처리됩니다.

    Examples
    --------
    >>> scores_norm = rank_normalize(flip_score(raw_scores))
    """
    scores = np.asarray(scores)
    return rankdata(scores, method='average') / len(scores)


# def smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
#     """
#     중앙 정렬 이동 평균으로 score를 스무딩합니다.

#     Point anomaly 주변의 급격한 score 변화를 완화하여
#     이상 구간 탐지 성능을 높입니다. window <= 1이면 그대로 반환합니다.

#     Examples
#     --------
#     >>> scores_smooth = smooth_scores(scores_norm, window=50)
#     """
#     if window <= 1:
#         return np.asarray(scores)
#     return pd.Series(scores).rolling(window, center=True, min_periods=1).mean().to_numpy()


def ensemble_scores(
    score_dict: dict[str, np.ndarray],
    weight_dict: dict[str, float] | None = None,
) -> np.ndarray:
    """
    여러 모델의 rank-normalized score를 앙상블합니다.

    Parameters
    ----------
    score_dict  : {'if': array, 'ocsvm': array, 'gmm': array} 형태
                  각 배열은 rank_normalize()가 적용된 상태여야 합니다.
    weight_dict : None이면 단순 평균, 전달하면 가중 평균
                  예) {'if': 0.5, 'ocsvm': 0.3, 'gmm': 0.2}

    Examples
    --------
    >>> final = ensemble_scores(
    ...     {'if': scores_if, 'gmm': scores_gmm},
    ...     weight_dict={'if': 0.6, 'gmm': 0.4},
    ... )
    """
    keys   = list(score_dict.keys())
    arrays = [np.asarray(score_dict[k]) for k in keys]

    if weight_dict is None:
        return np.mean(arrays, axis=0)

    total = sum(weight_dict[k] for k in keys)
    return sum(weight_dict[k] * np.asarray(score_dict[k]) for k in keys) / total
