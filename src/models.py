"""
이상징후 탐지 모델 모듈

이 모듈은 다음을 담당합니다:
  - IsolationForest, OneClassSVM, GaussianMixture 학습 및 score 추출

score 후처리, 앙상블, 평가는 ensemble.py에서 담당합니다.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM


def fit_isolation_forest(
    X_train: np.ndarray,
    X_val: np.ndarray,
    n_estimators: int = 100,
    contamination: float = 0.01,
    random_state: int = 42,
) -> tuple[IsolationForest, np.ndarray, np.ndarray]:
    """
    IsolationForest를 정상 데이터로 학습하고 anomaly score를 반환합니다.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
        학습용 feature matrix (정상 데이터만)
    X_val : np.ndarray, shape (n_val, n_features)
        검증용 feature matrix
    n_estimators : int, default 100
        트리 개수
    contamination : float, default 0.01
        학습 데이터 내 이상 비율 추정값 (정상 데이터만 사용하므로 작게 설정)
    random_state : int, default 42

    Returns
    -------
    model : IsolationForest
        학습된 모델
    score_train : np.ndarray
        학습 데이터 anomaly score (score_samples 원본값, 작을수록 이상)
    score_val : np.ndarray
        검증 데이터 anomaly score (score_samples 원본값, 작을수록 이상)

    Notes
    -----
    반환된 score는 작을수록 이상이므로, to_anomaly_score('if')로 방향 반전 필요.
    """
    # IF: contamination은 0에 가깝게 설정 (학습 자료: 정상 데이터만으로 학습)
    # 트리가 이상 샘플을 더 짧은 경로로 분리한다는 원리 활용
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)

    # IF: score_samples는 작을수록 이상 (학습자료: contamination은 0에 가깝게)
    score_train = model.score_samples(X_train)
    score_val = model.score_samples(X_val)

    return model, score_train, score_val


def fit_ocsvm(
    X_train: np.ndarray,
    X_val: np.ndarray,
    kernel: str = 'rbf',
    nu: float = 0.05,
    gamma: str | float = 'scale',
) -> tuple[OneClassSVM, np.ndarray, np.ndarray]:
    """
    OneClassSVM을 정상 데이터로 학습하고 anomaly score를 반환합니다.

    대용량 데이터(10,000행 초과)에서는 학습 속도를 위해 무작위 서브샘플링합니다.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
        학습용 feature matrix (정상 데이터만)
        10,000행 초과 시 10,000개로 서브샘플링하여 학습
    X_val : np.ndarray, shape (n_val, n_features)
        검증용 feature matrix (서브샘플링 없이 전체 사용)
    kernel : str, default 'rbf'
        커널 종류
    nu : float, default 0.05
        이상 비율 상한 및 서포트 벡터 비율 하한 (0 < nu <= 1)
    gamma : str or float, default 'scale'
        RBF 커널 파라미터

    Returns
    -------
    model : OneClassSVM
        학습된 모델
    score_train : np.ndarray
        학습 데이터 anomaly score (decision_function 원본값, 작을수록 이상)
    score_val : np.ndarray
        검증 데이터 anomaly score (decision_function 원본값, 작을수록 이상)

    Notes
    -----
    반환된 score는 작을수록 이상이므로, to_anomaly_score('ocsvm')로 방향 반전 필요.
    """
    # OCSVM: 대용량에서 학습이 O(n^2)로 느림 → 서브샘플링으로 속도 확보
    MAX_TRAIN_SAMPLES = 10_000
    X_fit = X_train
    if X_train.shape[0] > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(X_train.shape[0], size=MAX_TRAIN_SAMPLES, replace=False)
        X_fit = X_train[idx]

    model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    model.fit(X_fit)

    # OCSVM: f(x) = w·φ(x) - ρ, f(x) < 0 이면 이상
    # decision_function은 초평면까지의 부호 있는 거리 — 작을수록 이상
    score_train = model.decision_function(X_train)
    score_val = model.decision_function(X_val)

    return model, score_train, score_val


def fit_gmm(
    X_train: np.ndarray,
    X_val: np.ndarray,
    n_components: int = 3,
    covariance_type: str = 'full',
    random_state: int = 42,
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """
    GaussianMixture를 정상 데이터로 학습하고 anomaly score를 반환합니다.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
        학습용 feature matrix (정상 데이터만)
    X_val : np.ndarray, shape (n_val, n_features)
        검증용 feature matrix
    n_components : int, default 3
        가우시안 혼합 컴포넌트 수
    covariance_type : str, default 'full'
        공분산 행렬 형태 ('full', 'tied', 'diag', 'spherical')
    random_state : int, default 42

    Returns
    -------
    model : GaussianMixture
        학습된 모델
    score_train : np.ndarray
        학습 데이터 log-likelihood score (작을수록 이상)
    score_val : np.ndarray
        검증 데이터 log-likelihood score (작을수록 이상)

    Notes
    -----
    반환된 score는 작을수록 이상이므로, to_anomaly_score('gmm')로 방향 반전 필요.
    """
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    model.fit(X_train)

    # GMM: log p(x)가 낮을수록 이상 (정상 데이터 분포에서 멀수록)
    # score_samples는 각 샘플의 log-likelihood를 반환
    score_train = model.score_samples(X_train)
    score_val = model.score_samples(X_val)

    return model, score_train, score_val
