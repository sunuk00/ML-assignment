"""실험 005: GMM — x_f8 차분 + Rolling 통계 (W=30)

전처리: x_f8 차분 → 연속형 7채널 → StandardScaler → PCA(95%)
        → Rolling(W=30) → 5통계량(mean, std, min, max, range)
GMM(n_components=5, covariance_type='full')으로 확률 밀도를 추정합니다.

설계 근거:
002_rolling.py 기반에 x_f8 채널 차분(diff)을 추가합니다.
x_f8이 추세(drift)를 가진 채널이라면 차분으로 비정상성을 제거하여
GMM이 잔차 변화의 이상을 더 잘 포착할 수 있습니다.

피처 차원: x_f8 차분 → 연속형 7 → Scaler → PCA(95%) → Rolling W=30 → 각 PC × 5통계
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

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

DIFF_COL        = "x_f8"
WINDOW_SIZE     = 20
ROLL_STATS      = ['mean', 'std', 'min', 'max', 'range']
N_COMPONENTS    = 5
COVARIANCE_TYPE = "full"
PCA_VARIANCE    = 0.95
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)


def minmax(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def apply_diff(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """지정 채널을 1차 차분으로 교체합니다 (첫 번째 행은 0으로 채움)."""
    df = df.copy()
    df[col] = df[col].diff().fillna(0)
    return df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 005 GMM — x_f8 차분 + Rolling(W=30) ===\n")

    # 1. 데이터 로드
    train_df, _           = load_data("train",       str(DATA_DIR))
    val_df,   val_labels  = load_data("val",         str(DATA_DIR))
    test_df,  test_labels = load_data("test_public", str(DATA_DIR))
    print(f"  train {train_df.shape}  val {val_df.shape}  test {test_df.shape}")

    # 2. 결측치 처리
    train_df = fill_missing(train_df)
    val_df   = fill_missing(val_df)
    test_df  = fill_missing(test_df)

    # 3. x_f8 차분
    train_df = apply_diff(train_df, DIFF_COL)
    val_df   = apply_diff(val_df,   DIFF_COL)
    test_df  = apply_diff(test_df,  DIFF_COL)
    print(f"  {DIFF_COL} 차분 적용 완료")

    # 4. 연속형 채널만 사용
    x_cols    = [c for c in train_df.columns if c.startswith("x_")]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
    print(f"  연속형: {len(cont_cols)}채널 (이산형 제외)")

    # 5. StandardScaler (학습 데이터 기준)
    scaler       = fit_scaler(train_df[cont_cols])
    train_scaled = apply_scaler(scaler, train_df[cont_cols])
    val_scaled   = apply_scaler(scaler, val_df[cont_cols])
    test_scaled  = apply_scaler(scaler, test_df[cont_cols])

    # 6. PCA (학습 데이터 기준)
    pca       = fit_pca(train_scaled, variance=PCA_VARIANCE)
    train_pca = apply_pca(pca, train_scaled)
    val_pca   = apply_pca(pca, val_scaled)
    test_pca  = apply_pca(pca, test_scaled)
    print(f"  PCA components: {train_pca.shape[1]}  (variance={PCA_VARIANCE:.2f})")

    # 7. Rolling 통계 (PCA 성분 기준, W=30)
    pc_cols = list(train_pca.columns)
    train_X = rolling_features(train_pca, cols=pc_cols, window_size=WINDOW_SIZE, stats=ROLL_STATS).to_numpy()
    val_X   = rolling_features(val_pca,   cols=pc_cols, window_size=WINDOW_SIZE, stats=ROLL_STATS).to_numpy()
    test_X  = rolling_features(test_pca,  cols=pc_cols, window_size=WINDOW_SIZE, stats=ROLL_STATS).to_numpy()
    print(f"  feature dim: {train_X.shape[1]}  ({len(pc_cols)} PC × {len(ROLL_STATS)} 통계, W={WINDOW_SIZE})")

    # 8. GMM 학습
    print(f"  GMM n_components={N_COMPONENTS}  covariance_type={COVARIANCE_TYPE!r}")
    model = fit_gmm(train_X, n_components=N_COMPONENTS, covariance_type=COVARIANCE_TYPE)

    # 9. Score 계산
    val_scores  = minmax(-model.score_samples(val_X))
    test_scores = minmax(-model.score_samples(test_X))

    # 10. 평가
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

    # 11. 시각화
    plot_full(val_scores, val_labels, test_scores, test_labels,
              tag="005_gmm_diff_x_f8_rolling_w30", save_path=str(OUTPUT_DIR / "005_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="005_gmm_diff_x_f8_rolling_w30", save_path=str(OUTPUT_DIR / "005_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="005_gmm_diff_x_f8_rolling_w30", save_path=str(OUTPUT_DIR / "005_score_hist.png"))
