"""
이상탐지 최종 제출 코드
========================
모델   : LOF-Disc + GMM-003 앙상블
입력   : data/train.csv, data/test_hidden_no_labels.csv
출력   : test_hidden_no_labels_result.csv
실행   : python final_code.py

앙상블 전략
-----------
  각 모델 score를 부호 반전(flip) 후 rank/N 정규화 → 단순 평균(equal weight)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# 설정
# ============================================================

_BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_BASE, "data")
OUTPUT_FILE = os.path.join(_BASE, "test_hidden_no_labels_result.csv")

DISCRETE_COLS = ["x_06", "x_92", "x_4b"]   # 이산형(바이너리) 채널
DIFF_COL      = "x_f8"                      # 트렌드 제거용 차분 채널

# LOF-Disc 하이퍼파라미터
LOF_NEIGHBORS   = 10
LOF_CONTAM      = 0.0001
LOF_DISC_WINDOW = 10    # 이산형 rolling mean 윈도우

# GMM-003 하이퍼파라미터
GMM_N_COMP   = 5
GMM_COV_TYPE = "full"
GMM_PCA_VAR  = 0.95     # 누적 분산 95% 유지
GMM_WINDOW   = 20       # PCA 성분 rolling 윈도우
GMM_STATS    = ["mean", "std", "min", "max", "range"]


# ============================================================
# 1. 데이터 로드
# ============================================================

def load_split(name):
    """
    CSV 로드 → (feature DataFrame, labels | None)

    Parameters
    ----------
    name : "train" | "val" | "test_public" | "test_hidden_no_labels"

    Returns
    -------
    df     : t 컬럼 + x_ feature 컬럼 (label 분리됨)
    labels : 0/1 배열, 라벨 없으면 None
    """
    raw = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    if "label" in raw.columns:
        return raw.drop(columns=["label"]), raw["label"].to_numpy().astype(int)
    return raw, None


# ============================================================
# 2. 전처리
# ============================================================

def fill_missing(df):
    """선형 보간 → bfill → ffill."""
    if df.isnull().sum().sum() == 0:
        return df.copy()
    return df.copy().interpolate(method="linear").bfill().ffill()


def _apply_diff(df, col):
    """지정 채널 1-diff 적용 (트렌드 제거). 첫 행은 0으로 채움."""
    df = df.copy()
    df[col] = df[col].diff().fillna(0)
    return df


def fit_scaler(df):
    """train DataFrame으로 StandardScaler fit."""
    sc = StandardScaler()
    sc.fit(df.values)
    return sc


def apply_scaler(scaler, df):
    """fit된 scaler로 transform. 컬럼명·인덱스 유지."""
    return pd.DataFrame(
        scaler.transform(df.values), index=df.index, columns=df.columns
    )


def fit_pca(df, variance=0.95, random_state=42):
    """train DataFrame으로 PCA fit (분산 variance 이상의 주성분 수 자동 결정)."""
    pca = PCA(n_components=variance, random_state=random_state)
    pca.fit(df.values)
    return pca


def apply_pca(pca, df):
    """fit된 PCA로 transform. 컬럼명: pc_01, pc_02, ..."""
    t = pca.transform(df.values)
    return pd.DataFrame(
        t, index=df.index, columns=[f"pc_{i+1:02d}" for i in range(t.shape[1])]
    )


def _rolling_stats(df, cols, window, stats):
    """
    pandas rolling 통계량 계산 (min_periods=1, 행 수 보존).

    Parameters
    ----------
    df     : 입력 DataFrame
    cols   : 통계를 계산할 컬럼 리스트
    window : rolling 윈도우 크기
    stats  : ["mean", "std", "min", "max", "range"] 부분집합

    Returns
    -------
    pd.DataFrame  컬럼명 '{col}_{stat}', 행 수 = 입력과 동일
    """
    parts = {}
    for col in cols:
        s = pd.Series(df[col].values, index=df.index)
        r = s.rolling(window=window, min_periods=1)
        _mx = _mn = None
        for stat in stats:
            if stat == "mean":
                parts[f"{col}_mean"] = r.mean().to_numpy()
            elif stat == "std":
                parts[f"{col}_std"] = r.std().fillna(0).to_numpy()
            elif stat == "min":
                _mn = r.min().to_numpy()
                parts[f"{col}_min"] = _mn
            elif stat == "max":
                _mx = r.max().to_numpy()
                parts[f"{col}_max"] = _mx
            elif stat == "range":
                if _mx is None: _mx = r.max().to_numpy()
                if _mn is None: _mn = r.min().to_numpy()
                parts[f"{col}_range"] = _mx - _mn
    return pd.DataFrame(parts, index=df.index).fillna(0)


# ============================================================
# 3. Score 정규화 / 앙상블
# ============================================================

def flip_score(scores):
    """'작을수록 이상' → '클수록 이상' 부호 반전."""
    return -1.0 * np.asarray(scores)


def rank_normalize(scores):
    """rank / N 정규화 → [0, 1]. 동점은 평균 rank."""
    scores = np.asarray(scores)
    return rankdata(scores, method="average") / len(scores)


def ensemble_mean(score_dict):
    """rank 정규화된 score 단순 평균 (equal weight)."""
    return np.mean(list(score_dict.values()), axis=0)


# ============================================================
# 4. LOF_Disc 피처 빌더
# ============================================================

def build_lof_disc_features(train_df, infer_df, cont_cols, disc_cols):
    """
    LOF_Disc 파이프라인:
      x_f8 차분 → 연속형 7ch StandardScaler
               + 이산형 3ch rolling mean(W=LOF_DISC_WINDOW)
      → (train_X, infer_X)  feature dim: 10
    """
    tr = _apply_diff(train_df, DIFF_COL)
    te = _apply_diff(infer_df, DIFF_COL)

    scaler    = fit_scaler(tr[cont_cols])
    tr_scaled = apply_scaler(scaler, tr[cont_cols]).to_numpy()
    te_scaled = apply_scaler(scaler, te[cont_cols]).to_numpy()

    def _disc_ratio(df):
        return (df[disc_cols]
                .rolling(LOF_DISC_WINDOW, min_periods=1)
                .mean().fillna(0).to_numpy())

    return (
        np.hstack([tr_scaled, _disc_ratio(tr)]),
        np.hstack([te_scaled, _disc_ratio(te)]),
    )


# ============================================================
# 5. GMM-003 피처 빌더
# ============================================================

def build_gmm_features(train_df, infer_df, cont_cols):
    """
    GMM-003 파이프라인:
      x_f8 차분 → StandardScaler → PCA(95%)
               → Rolling(W=20, mean/std/min/max/range)
      → (train_X, infer_X)
    """
    tr = _apply_diff(train_df, DIFF_COL)
    te = _apply_diff(infer_df, DIFF_COL)

    scaler    = fit_scaler(tr[cont_cols])
    tr_scaled = apply_scaler(scaler, tr[cont_cols])
    te_scaled = apply_scaler(scaler, te[cont_cols])

    pca    = fit_pca(tr_scaled, variance=GMM_PCA_VAR)
    tr_pca = apply_pca(pca, tr_scaled)
    te_pca = apply_pca(pca, te_scaled)

    pc_cols = list(tr_pca.columns)
    tr_X = _rolling_stats(tr_pca, pc_cols, GMM_WINDOW, GMM_STATS).to_numpy()
    te_X = _rolling_stats(te_pca, pc_cols, GMM_WINDOW, GMM_STATS).to_numpy()
    return tr_X, te_X


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    print("=== 최종 제출: LOF_Disc + GMM-003 앙상블 ===\n")

    # ---------- 데이터 로드 ----------
    train_df,  _ = load_split("train")
    hidden_df, _ = load_split("test_hidden_no_labels")
    print(f"train:  {train_df.shape}")
    print(f"hidden: {hidden_df.shape}")

    # ---------- 결측치 처리 ----------
    train_df  = fill_missing(train_df)
    hidden_df = fill_missing(hidden_df)

    # ---------- 채널 분리 ----------
    x_cols    = [c for c in train_df.columns if c.startswith("x_")]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
    disc_cols = [c for c in x_cols if c in     DISCRETE_COLS]
    print(f"연속형 {len(cont_cols)}채널  이산형 {len(disc_cols)}채널")

    # ---------- LOF-Disc ----------
    print("\nLOF-Disc 피처 구성 및 학습 중...")
    lof_train_X, lof_hidden_X = build_lof_disc_features(
        train_df, hidden_df, cont_cols, disc_cols
    )
    lof_model = LocalOutlierFactor(
        n_neighbors=LOF_NEIGHBORS,
        contamination=LOF_CONTAM,
        novelty=True,
        n_jobs=-1,
    )
    lof_model.fit(lof_train_X)
    lof_hidden = rank_normalize(flip_score(lof_model.score_samples(lof_hidden_X)))
    print(f"  feature dim: {lof_train_X.shape[1]}")

    # ---------- GMM-003 ----------
    print("GMM-003 피처 구성 및 학습 중...")
    gmm_train_X, gmm_hidden_X = build_gmm_features(
        train_df, hidden_df, cont_cols
    )
    gmm_model = GaussianMixture(
        n_components=GMM_N_COMP,
        covariance_type=GMM_COV_TYPE,
        random_state=42,
    )
    gmm_model.fit(gmm_train_X)
    gmm_hidden = rank_normalize(flip_score(gmm_model.score_samples(gmm_hidden_X)))
    print(f"  feature dim: {gmm_train_X.shape[1]}")

    # ---------- 앙상블 ----------
    ens_scores = ensemble_mean({"lof": lof_hidden, "gmm": gmm_hidden})

    # ---------- CSV 저장 ----------
    t_vals = (hidden_df["t"].to_numpy()
              if "t" in hidden_df.columns
              else np.arange(len(hidden_df)))
    pd.DataFrame({"t": t_vals, "score": ens_scores}).to_csv(OUTPUT_FILE, index=False)
    print(f"\n저장 완료: {OUTPUT_FILE}")
    print(f"  timestep 수: {len(ens_scores):,}  |  "
          f"score 범위: [{ens_scores.min():.4f}, {ens_scores.max():.4f}]")
