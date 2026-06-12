"""실험 004: GMM — PC1·PC2만 선택 + Rolling + Skewness/Kurtosis

연속형 채널(7개)에 StandardScaler → PCA 후 PC1·PC2만 선택하고, Rolling 통계(mean/std/min/max/range/skew/kurt)를 적용합니다.
GaussianMixture로 학습 및 추론합니다.

피처 차원: 연속형 7 → StandardScaler → PCA → PC1·PC2 × 7통계
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader   import load_data
from src.preprocessing import fill_missing, fit_scaler, apply_scaler, fit_pca, apply_pca
from src.features      import DISCRETE_COLS, rolling_features
from src.models        import fit_gmm
from src.ensemble      import flip_score, rank_normalize
from src.evaluate      import evaluate_aupr, evaluate_auroc, anomaly_type_aupr, plot_full, plot_zooms, plot_score_hist

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "gmm" / "outputs"

WINDOW_SIZE     = 20
ROLL_STATS      = ['mean', 'std', 'min', 'max', 'range', 'skew', 'kurt']
N_PC_KEEP       = 2          # PC1·PC2만 사용
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
    print("=== 004 GMM — PC1·PC2 only + Rolling(W=200) + Skew/Kurt ===\n")

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

    # 5. PCA (학습 데이터 기준) → 상위 N_PC_KEEP개 성분만 선택
    pca       = fit_pca(train_scaled, variance=PCA_VARIANCE)
    train_pca = apply_pca(pca, train_scaled)
    val_pca   = apply_pca(pca, val_scaled)
    test_pca  = apply_pca(pca, test_scaled)

    pc_cols = list(train_pca.columns)[:N_PC_KEEP]   # ['pc_01', 'pc_02']
    train_pca = train_pca[pc_cols]
    val_pca   = val_pca[pc_cols]
    test_pca  = test_pca[pc_cols]
    print(f"  PCA 전체 components: {pca.n_components_}  → 상위 {N_PC_KEEP}개만 사용: {pc_cols}")

    # 6. Rolling 통계 (PC1·PC2 기준, W=200, skew/kurt 포함)
    train_X = rolling_features(train_pca, cols=pc_cols, window_size=WINDOW_SIZE, stats=ROLL_STATS).to_numpy()
    val_X   = rolling_features(val_pca,   cols=pc_cols, window_size=WINDOW_SIZE, stats=ROLL_STATS).to_numpy()
    test_X  = rolling_features(test_pca,  cols=pc_cols, window_size=WINDOW_SIZE, stats=ROLL_STATS).to_numpy()
    print(f"  feature dim: {train_X.shape[1]}  ({N_PC_KEEP} PC × {len(ROLL_STATS)} 통계, W={WINDOW_SIZE})")

    # 7. GMM 학습
    print(f"  GMM n_components={N_COMPONENTS}  covariance_type={COVARIANCE_TYPE!r}")
    model = fit_gmm(train_X, n_components=N_COMPONENTS, covariance_type=COVARIANCE_TYPE)

    # 8. Score 계산 (log-likelihood → 작을수록 이상 → flip)
    # val_scores  = rank_normalize(flip_score(model.score_samples(val_X)))
    # test_scores = rank_normalize(flip_score(model.score_samples(test_X)))
    
    # min-max 정규화 버전 (LOF는 이상치 점수가 클수록 정상에 가깝기 때문에 flip_score → -score_samples)
    val_scores  = minmax(-model.score_samples(val_X))
    test_scores = minmax(-model.score_samples(test_X))

    # 9. 평가
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

    # 10. 시각화
    plot_full(val_scores, val_labels, test_scores, test_labels,
              tag="004_gmm_pc2_rolling_w200_skew_kurt", save_path=str(OUTPUT_DIR / "004_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="004_gmm_pc2_rolling_w200_skew_kurt", save_path=str(OUTPUT_DIR / "004_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="004_gmm_pc2_rolling_w200_skew_kurt", save_path=str(OUTPUT_DIR / "004_score_hist.png"))
