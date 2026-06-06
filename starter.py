"""
이상탐지 과제 시작 코드 (Starter Code)
=========================================

이 파일은 가장 단순한 baseline 파이프라인을 보여줍니다:
  1. 데이터 로드
  2. 전처리 (StandardScaler)
  3. Sliding window 변환
  4. 모델 학습 (Isolation Forest)
  5. test_public으로 평가 (AUROC, AUPR)

이 baseline을 출발점으로 삼아 본인의 모델/전처리로 발전시키세요.
어디를 수정하면 좋을지는 main 함수 안에 주석으로 표시되어 있습니다.

주의사항:
  - train.csv는 정상 데이터만 포함합니다 (label 컬럼 없음).
  - val.csv는 하이퍼파라미터 튜닝용입니다.
  - test_public.csv는 자체 성능 검증용입니다.
  - test_hidden_no_labels.csv는 최종 제출용이며 라벨이 없습니다.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score


# 데이터 디렉토리. 본인 환경에 맞게 수정하세요.
DATA_DIR = "./data"


# ============================================================
# 1. 데이터 로드
# ============================================================

def load_split(name, data_dir=DATA_DIR):
    """
    하나의 split CSV를 로드합니다.

    Parameters
    ----------
    name : str
        "train", "val", "test_public", "test_hidden_no_labels" 중 하나.
    data_dir : str
        CSV들이 있는 디렉토리 경로.

    Returns
    -------
    df : pd.DataFrame
        timestep + feature 컬럼 (label은 분리되어 있음)
    feature_cols : list[str]
        feature 컬럼 이름들 (x_로 시작하는 것들)
    labels : np.ndarray | None
        timestep별 라벨 (0=정상, 1=이상). 라벨이 없는 split이면 None.
    """
    path = os.path.join(data_dir, f"{name}.csv")
    raw = pd.read_csv(path)

    feature_cols = [c for c in raw.columns if c.startswith("x_")]

    if "label" in raw.columns:
        labels = raw["label"].to_numpy().astype(int)
        df = raw.drop(columns=["label"])
    else:
        labels = None
        df = raw

    return df, feature_cols, labels


# ============================================================
# 2. Sliding window
# ============================================================

def make_windows(values, window_size, stride=1):
    """
    시계열을 sliding window로 변환합니다.

    Parameters
    ----------
    values : np.ndarray
        shape (T,) 또는 (T, D)
    window_size : int
    stride : int

    Returns
    -------
    windows : np.ndarray
        - 입력이 (T,)면 출력은 (N, window_size)
        - 입력이 (T, D)면 출력은 (N, window_size, D)
        N = (T - window_size) // stride + 1
    """
    values = np.asarray(values)
    T = values.shape[0]
    if T < window_size:
        raise ValueError(f"입력 길이({T})가 window_size({window_size})보다 짧습니다.")

    n = (T - window_size) // stride + 1
    if values.ndim == 1:
        out = np.stack([values[i*stride : i*stride + window_size]
                        for i in range(n)])
    elif values.ndim == 2:
        out = np.stack([values[i*stride : i*stride + window_size, :]
                        for i in range(n)])
    else:
        raise ValueError(f"지원하지 않는 차원: {values.ndim}")
    return out


def windows_to_timestep_scores(window_scores, T, window_size, stride=1):
    """
    window별 score를 timestep별 score로 환산합니다.

    가장 단순한 방식: window의 score를 그 window의 마지막 timestep에 할당.
    첫 (window_size - 1) timestep은 첫 window의 score로 패딩.
    중간에 빈 timestep이 있으면 forward-fill로 채움.

    이 변환 방식은 baseline일 뿐입니다. 더 나은 방식 (예: window 중심에 할당,
    겹치는 window들의 평균 등)을 직접 구현해보세요.

    Parameters
    ----------
    window_scores : np.ndarray, shape (N,)
        각 window의 anomaly score
    T : int
        원래 시계열 길이
    window_size : int
    stride : int

    Returns
    -------
    timestep_scores : np.ndarray, shape (T,)
    """
    timestep_scores = np.full(T, np.nan)
    n_windows = len(window_scores)

    # 각 window의 score를 그 window의 마지막 timestep에 할당
    for i in range(n_windows):
        end_idx = i * stride + window_size - 1
        timestep_scores[end_idx] = window_scores[i]

    # 앞쪽 패딩 (첫 window가 끝나기 전 구간)
    timestep_scores[:window_size - 1] = window_scores[0]

    # stride > 1인 경우 중간에 nan이 남을 수 있어 forward-fill
    for i in range(1, T):
        if np.isnan(timestep_scores[i]):
            timestep_scores[i] = timestep_scores[i - 1]

    return timestep_scores


# ============================================================
# 3. Baseline 파이프라인
# ============================================================

if __name__ == "__main__":
    # ---------- 데이터 로드 ----------
    train_df, feature_cols, _ = load_split("train")
    val_df,   _, val_labels   = load_split("val")
    test_df,  _, test_labels  = load_split("test_public")

    print("=== 데이터 형태 ===")
    print(f"train:        {train_df.shape}, anomaly=없음 (정상만)")
    print(f"val:          {val_df.shape}, anomaly={val_labels.sum()}개 timestep")
    print(f"test_public:  {test_df.shape}, anomaly={test_labels.sum()}개 timestep")
    print(f"feature_cols: {feature_cols}")
    print()

    # ---------- 전처리: 스케일링 ----------
    # train으로만 fit, val/test에는 transform만 적용 (data leakage 방지)
    # ※ 개선 포인트: 연속형/이산형을 분리해서 다르게 처리, RobustScaler 시도, 등
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val   = scaler.transform(val_df[feature_cols])
    X_test  = scaler.transform(test_df[feature_cols])

    # ---------- Sliding window ----------
    # ※ 개선 포인트: window 크기 튜닝, 통계 피처(mean/std/min/max) 추출, 등
    W = 50   # window 크기
    S = 1    # stride

    train_windows = make_windows(X_train, W, S)  # (N, W, D)
    val_windows   = make_windows(X_val,   W, S)
    test_windows  = make_windows(X_test,  W, S)

    # IsolationForest는 1D 입력을 기대하므로 (N, W, D) -> (N, W*D)로 flatten
    # ※ 개선 포인트: flatten 대신 window별 통계량 추출이 더 나을 수 있음
    train_X = train_windows.reshape(len(train_windows), -1)
    val_X   = val_windows.reshape(len(val_windows), -1)
    test_X  = test_windows.reshape(len(test_windows), -1)

    print(f"=== Sliding window (W={W}, stride={S}) ===")
    print(f"train_X: {train_X.shape}")
    print(f"val_X:   {val_X.shape}")
    print(f"test_X:  {test_X.shape}")

    # ---------- 모델 학습: Isolation Forest ----------
    # ※ 개선 포인트:
    #   - 다른 모델 시도 (One-Class SVM, LOF, GMM, PCA-based 등)
    #   - n_estimators, max_samples, max_features 등 HP 튜닝 (val로)
    print("=== Isolation Forest 학습 중 ===")
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_X)
    print("학습 완료")

    # ---------- score 계산 ----------
    # IsolationForest.score_samples는 "정상일수록 큰 값"을 반환하므로
    # anomaly score로 쓰려면 부호를 뒤집음 (-)
    val_window_scores  = -model.score_samples(val_X)
    test_window_scores = -model.score_samples(test_X)

    # window score → timestep score 환산
    val_scores  = windows_to_timestep_scores(val_window_scores,  len(val_df),  W, S)
    test_scores = windows_to_timestep_scores(test_window_scores, len(test_df), W, S)

    # ---------- 평가 ----------
    val_auroc  = roc_auc_score(val_labels,  val_scores)
    val_aupr   = average_precision_score(val_labels,  val_scores)
    test_auroc = roc_auc_score(test_labels, test_scores)
    test_aupr  = average_precision_score(test_labels, test_scores)

    print("=== Baseline 성능 ===")
    print(f"{'':12s} {'AUROC':>8s} {'AUPR':>8s}")
    print(f"{'val':12s} {val_auroc:>8.4f} {val_aupr:>8.4f}")
    print(f"{'test_public':12s} {test_auroc:>8.4f} {test_aupr:>8.4f}")
    print()
    print("이 baseline을 출발점으로 본인의 모델/전처리/피처로 개선해보세요.")

    # =========================================================
    # - test_hidden_no_labels.csv에 대한 anomaly score 생성
    # - (t, score) 두 컬럼의 CSV로 저장하여 제출
    # =========================================================