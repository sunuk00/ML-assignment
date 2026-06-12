"""실험 001: GMM — Baseline (연속형 + StandardScaler + PCA)

연속형 채널(7개)에 StandardScaler → PCA를 적용합니다.
슬라이딩 윈도우 없이 단일 시점 데이터를 사용합니다.
GaussianMixture로 학습 및 추론합니다.

피처 차원: 연속형 7(raw) → StandardScaler → PCA
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader   import load_data
from src.preprocessing import fill_missing, fit_scaler, apply_scaler, fit_pca, apply_pca
from src.features      import DISCRETE_COLS
from src.models        import fit_gmm
from src.ensemble      import flip_score, rank_normalize
from src.evaluate      import evaluate_aupr, evaluate_auroc, anomaly_type_aupr, plot_full, plot_zooms, plot_score_hist

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "gmm" / "outputs"

N_COMPONENTS    = 5
COVARIANCE_TYPE = "full"
PCA_VARIANCE    = 0.95
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)

def minmax(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 001 GMM — Baseline (단일 시점, Scaler + PCA) ===\n")

    # 1. 데이터 로드
    train_df, _           = load_data("train",       str(DATA_DIR))
    val_df,   val_labels  = load_data("val",         str(DATA_DIR))
    test_df,  test_labels = load_data("test_public", str(DATA_DIR))
    print(f"  train {train_df.shape}  val {val_df.shape}  test {test_df.shape}")

    # 2. 결측치 처리
    train_df = fill_missing(train_df)
    val_df   = fill_missing(val_df)
    test_df  = fill_missing(test_df)

    # 3. 연속형 채널만 사용
    x_cols    = [c for c in train_df.columns if c.startswith("x_")]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
    print(f"  연속형: {len(cont_cols)}채널 (이산형 제외)")

    # 4. StandardScaler (학습 데이터 기준)
    scaler       = fit_scaler(train_df[cont_cols])
    train_scaled = apply_scaler(scaler, train_df[cont_cols])
    val_scaled   = apply_scaler(scaler, val_df[cont_cols])
    test_scaled  = apply_scaler(scaler, test_df[cont_cols])

    # 5. PCA (학습 데이터 기준)
    pca     = fit_pca(train_scaled, variance=PCA_VARIANCE)
    train_X = apply_pca(pca, train_scaled).to_numpy()
    val_X   = apply_pca(pca, val_scaled).to_numpy()
    test_X  = apply_pca(pca, test_scaled).to_numpy()
    print(f"  PCA components: {train_X.shape[1]}  (variance={PCA_VARIANCE:.2f})")

    # 6. GMM 학습
    print(f"  GMM n_components={N_COMPONENTS}  covariance_type={COVARIANCE_TYPE!r}")
    model = fit_gmm(train_X, n_components=N_COMPONENTS, covariance_type=COVARIANCE_TYPE)

    # 7. Score 계산 (log-likelihood → 작을수록 이상 → flip)
    # val_scores  = rank_normalize(flip_score(model.score_samples(val_X)))
    # test_scores = rank_normalize(flip_score(model.score_samples(test_X)))
    
    # min-max 정규화 버전 (LOF는 이상치 점수가 클수록 정상에 가깝기 때문에 flip_score → -score_samples)
    val_scores  = minmax(-model.score_samples(val_X))
    test_scores = minmax(-model.score_samples(test_X))

    # 8. 평가
    print(f"\n  val  AUROC={evaluate_auroc(val_scores, val_labels):.4f}  AUPR={evaluate_aupr(val_scores, val_labels):.4f}")
    print(f"  test AUROC={evaluate_auroc(test_scores, test_labels):.4f}  AUPR={evaluate_aupr(test_scores, test_labels):.4f}")
    print(f"\n  [Val] 유형별 AUPR")
    print(f"  Point      {anomaly_type_aupr(val_scores, val_labels, *POINT_LEN):.4f}")
    print(f"  Contextual {anomaly_type_aupr(val_scores, val_labels, *CONTEXTUAL_LEN):.4f}")
    print(f"  Collective {anomaly_type_aupr(val_scores, val_labels, *COLLECTIVE_LEN):.4f}")
    print(f"\n  [Test] 유형별 AUPR")
    print(f"  Point      {anomaly_type_aupr(test_scores, test_labels, *POINT_LEN):.4f}")
    print(f"  Contextual {anomaly_type_aupr(test_scores, test_labels, *CONTEXTUAL_LEN):.4f}")
    print(f"  Collective {anomaly_type_aupr(test_scores, test_labels, *COLLECTIVE_LEN):.4f}")

    # 9. 시각화
    plot_full(val_scores, val_labels, test_scores, test_labels,
              tag="001_gmm_baseline", save_path=str(OUTPUT_DIR / "001_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="001_gmm_baseline", save_path=str(OUTPUT_DIR / "001_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="001_gmm_baseline", save_path=str(OUTPUT_DIR / "001_score_hist.png"))
