from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def fit_isolation_forest(
    X_train: np.ndarray,
    n_estimators: int = 100,
    contamination: float = 0.01,
    random_state: int = 42,
) -> IsolationForest:
    """
    IsolationForest를 학습합니다.

    score 추출은 model.score_samples(X)로 직접 호출하세요.
    반환값이 작을수록 이상 → flip_score()로 방향 반전 필요.

    Examples
    --------
    >>> model       = fit_isolation_forest(X_train, n_estimators=200)
    >>> raw_scores  = model.score_samples(X_val)   # 작을수록 이상
    >>> scores      = flip_score(raw_scores)        # 클수록 이상
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def fit_ocsvm(
    X_train: np.ndarray,
    kernel: str = 'rbf',
    nu: float = 0.05,
    gamma: str | float = 'scale',
) -> OneClassSVM:
    """
    OneClassSVM을 학습합니다.

    10,000행 초과 시 학습 속도를 위해 자동 서브샘플링합니다.
    score 추출은 model.decision_function(X)로 직접 호출하세요.
    반환값이 작을수록 이상 → flip_score()로 방향 반전 필요.

    Examples
    --------
    >>> model       = fit_ocsvm(X_train, nu=0.05)
    >>> raw_scores  = model.decision_function(X_val)  # 작을수록 이상
    >>> scores      = flip_score(raw_scores)           # 클수록 이상
    """
    X_fit = X_train
    if X_train.shape[0] > 10_000:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(X_train.shape[0], size=10_000, replace=False)
        X_fit = X_train[idx]

    model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    model.fit(X_fit)
    return model


def fit_gmm(
    X_train: np.ndarray,
    n_components: int = 3,
    covariance_type: str = 'full',
    random_state: int = 42,
) -> GaussianMixture:
    """
    GaussianMixture를 학습합니다.

    score 추출은 model.score_samples(X)로 직접 호출하세요.
    반환값(log-likelihood)이 작을수록 이상 → flip_score()로 방향 반전 필요.

    Examples
    --------
    >>> model       = fit_gmm(X_train, n_components=5)
    >>> raw_scores  = model.score_samples(X_val)  # 작을수록 이상
    >>> scores      = flip_score(raw_scores)       # 클수록 이상
    """
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    model.fit(X_train)
    return model


def fit_lof(
    X_train: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.01,
    metric: str = 'minkowski',
) -> LocalOutlierFactor:
    """
    LocalOutlierFactor를 학습합니다 (novelty=True).

    score 추출은 model.score_samples(X)로 직접 호출하세요.
    반환값(음수 LOF 점수)이 작을수록 이상 → flip_score()로 방향 반전 필요.

    Examples
    --------
    >>> model       = fit_lof(X_train, n_neighbors=20)
    >>> raw_scores  = model.score_samples(X_val)  # 작을수록 이상
    >>> scores      = flip_score(raw_scores)       # 클수록 이상
    """
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        metric=metric,
        novelty=True,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model
