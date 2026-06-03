"""
앙상블 및 평가 모듈

이 모듈은 다음을 담당합니다:
  - Anomaly score 방향 통일 (작을수록 이상 → 클수록 이상)
  - Rank 정규화 (0~1)
  - 앙상블 score 계산 (단순 평균 / 가중 평균)
  - 평가 지표 계산 (AUPR, AUROC)

모델 학습은 models.py에서 담당합니다.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def to_anomaly_score(scores: np.ndarray, model_type: str) -> np.ndarray:
    """
    모델별 원본 score를 "클수록 이상" 방향으로 통일합니다.

    IF, OCSVM, GMM 모두 원본 score가 작을수록 이상이므로
    -1을 곱하여 방향을 반전합니다.

    Parameters
    ----------
    scores : np.ndarray
        모델에서 추출한 원본 score 배열
    model_type : str
        모델 종류 ('if', 'ocsvm', 'gmm')

    Returns
    -------
    np.ndarray
        "클수록 이상" 방향으로 변환된 score 배열

    Raises
    ------
    ValueError
        model_type이 지원되지 않는 값인 경우
    """
    supported = {'if', 'ocsvm', 'gmm'}
    if model_type not in supported:
        raise ValueError(
            f"model_type은 {supported} 중 하나여야 합니다. 입력값: {model_type!r}"
        )

    # IF, OCSVM, GMM 모두 원본 score가 작을수록 이상
    # → -1 곱해 방향 반전: 이후 rank_normalize에서 scale 통일
    return -1.0 * scores


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Score 배열을 rank 기반으로 0~1 사이로 정규화합니다.

    scipy.stats.rankdata로 rank를 계산한 뒤 샘플 수로 나눕니다.

    Parameters
    ----------
    scores : np.ndarray
        정규화할 score 배열 (1-D)

    Returns
    -------
    np.ndarray
        0~1 범위로 정규화된 score 배열

    Notes
    -----
    Rank 정규화: IF는 0~1 근방, OCSVM은 부호 있는 거리값 → scale 통일 필수.
    동점(tie)은 평균 rank로 처리됩니다(rankdata 기본값 'average').
    """
    # 모델마다 score의 분포·scale이 달라 직접 평균 불가 → rank로 통일
    n = len(scores)
    ranks = rankdata(scores, method='average')
    return ranks / n


def ensemble_scores(
    score_dict: dict[str, np.ndarray],
    weight_dict: dict[str, float] | None = None,
) -> np.ndarray:
    """
    여러 모델의 rank-normalized score를 앙상블하여 최종 score를 계산합니다.

    Parameters
    ----------
    score_dict : dict[str, np.ndarray]
        모델 이름을 key로 하는 rank-normalized score 딕셔너리
        예) {'if': array, 'ocsvm': array, 'gmm': array}
        입력 score들은 rank_normalize()가 적용된 상태여야 합니다.
    weight_dict : dict[str, float] or None, default None
        각 모델의 가중치 딕셔너리.
        None이면 단순 평균, 전달하면 가중 평균을 계산합니다.
        score_dict의 key와 동일한 key를 가져야 합니다.

    Returns
    -------
    np.ndarray
        앙상블된 최종 score 배열

    Warns
    -----
    두 모델 간 Pearson 상관계수가 0.95를 초과하면 경고를 출력합니다.
    (중복 정보가 많아 앙상블 다양성이 낮을 수 있음)
    """
    keys = list(score_dict.keys())
    arrays = [score_dict[k] for k in keys]

    # Pearson 상관 경고: 두 모델 score가 너무 유사하면 앙상블 효과 감소
    CORR_THRESHOLD = 0.95
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            corr = np.corrcoef(arrays[i], arrays[j])[0, 1]
            if corr > CORR_THRESHOLD:
                warnings.warn(
                    f"[ensemble] '{keys[i]}'와 '{keys[j]}' score의 Pearson 상관계수 "
                    f"({corr:.4f})가 {CORR_THRESHOLD}를 초과합니다. "
                    "두 모델이 유사한 패턴을 학습했을 수 있습니다.",
                    UserWarning,
                    stacklevel=2,
                )

    if weight_dict is None:
        # 단순 평균: 모든 모델 동등 기여
        return np.mean(arrays, axis=0)

    # 가중 평균: weight_dict에 지정된 비율로 합산
    total_weight = sum(weight_dict[k] for k in keys)
    weighted_sum = np.zeros_like(arrays[0], dtype=float)
    for k, arr in zip(keys, arrays):
        weighted_sum += weight_dict[k] * arr
    return weighted_sum / total_weight


def evaluate_aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUPR (Area Under Precision-Recall Curve)을 계산합니다.

    Parameters
    ----------
    scores : np.ndarray
        예측 anomaly score 배열 (클수록 이상)
    labels : np.ndarray
        정답 레이블 배열 (1=이상, 0=정상)

    Returns
    -------
    float
        AUPR 값 (0~1, 높을수록 좋음)
    """
    return float(average_precision_score(labels, scores))


def evaluate_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUROC (Area Under ROC Curve)을 계산합니다.

    Parameters
    ----------
    scores : np.ndarray
        예측 anomaly score 배열 (클수록 이상)
    labels : np.ndarray
        정답 레이블 배열 (1=이상, 0=정상)

    Returns
    -------
    float
        AUROC 값 (0~1, 높을수록 좋음)
    """
    return float(roc_auc_score(labels, scores))
