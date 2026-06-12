"""실험 010: LOF — x_f8 차분 + 연속형 윈도우 통계 + StandardScaler

x_f8 채널에만 1차 차분(diff)을 적용하여 장기 트렌드를 제거한 뒤,
연속형 7채널에 대해 슬라이딩 윈도우(W=50) 통계(평균/표준편차/최소/최대/중앙값)를
계산하고 학습 데이터 기준 StandardScaler를 적용하여 LOF로 이상치 점수를 산출합니다.

피처 차원: 연속형 7채널 × 5통계 = 35
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader   import load_data
from src.preprocessing import fill_missing, fit_scaler, apply_scaler
from src.features      import DISCRETE_COLS, rolling_features
from src.models        import fit_lof
from src.ensemble      import flip_score, rank_normalize
from src.evaluate      import evaluate_aupr, evaluate_auroc, anomaly_type_aupr, plot_full, plot_score_hist, plot_zooms

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "lof" / "outputs"

N_NEIGHBORS   = 10
CONTAMINATION = 0.0001
WINDOW_SIZE   = 5
DIFF_COL      = "x_f8"
STATS         = ["mean", "std", "min", "max", "median"]
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)


def minmax(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def _apply_diff(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].diff(periods=1).fillna(0)
    return df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 010 LOF — x_f8 Diff + Window Stats(W=50) + StandardScaler ===\n")

    # 1. 데이터 로드
    train_df, _           = load_data("train",       str(DATA_DIR))
    val_df,   val_labels  = load_data("val",         str(DATA_DIR))
    test_df,  test_labels = load_data("test_public", str(DATA_DIR))
    print(f"  train {train_df.shape}  val {val_df.shape}  test {test_df.shape}")

    # 2. 결측치 처리
    train_df = fill_missing(train_df)
    val_df   = fill_missing(val_df)
    test_df  = fill_missing(test_df)

    # 3. x_f8 차분 적용 (트렌드 제거)
    train_df = _apply_diff(train_df, DIFF_COL)
    val_df   = _apply_diff(val_df,   DIFF_COL)
    test_df  = _apply_diff(test_df,  DIFF_COL)

    # 4. 연속형 채널만 사용
    x_cols    = [c for c in train_df.columns if c.startswith("x_")]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
    print(f"  연속형: {len(cont_cols)}채널 (이산형 제외, x_f8 diff 적용됨)")

    # 5. 윈도우 통계 계산
    train_feats = rolling_features(train_df, cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)
    val_feats   = rolling_features(val_df,   cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)
    test_feats  = rolling_features(test_df,  cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)

    # 6. StandardScaler (train 기준)
    scaler       = fit_scaler(train_feats)
    train_scaled = apply_scaler(scaler, train_feats)
    val_scaled   = apply_scaler(scaler, val_feats)
    test_scaled  = apply_scaler(scaler, test_feats)

    # 7. 피처 준비
    train_X = train_scaled.to_numpy()
    val_X   = val_scaled.to_numpy()
    test_X  = test_scaled.to_numpy()
    print(f"  feature dim: {train_X.shape[1]}  (cont {len(cont_cols)}×{len(STATS)} window stats, W={WINDOW_SIZE} + scaler)")

    # 8. LOF 학습
    model = fit_lof(train_X, n_neighbors=N_NEIGHBORS, contamination=CONTAMINATION)

    # 9. Score 계산
    val_scores  = minmax(-model.score_samples(val_X))
    test_scores = minmax(-model.score_samples(test_X))

    # 10. 평가 출력
    print(f"\n  val  AUROC={evaluate_auroc(val_scores, val_labels):.4f}  AUPR={evaluate_aupr(val_scores, val_labels):.4f}")
    print(f"  test AUROC={evaluate_auroc(test_scores, test_labels):.4f}  AUPR={evaluate_aupr(test_scores, test_labels):.4f}")
    print("\n  [Val] 유형별 AUPR")
    print(f"  Point      {anomaly_type_aupr(val_scores, val_labels, *POINT_LEN):.4f}")
    print(f"  Contextual {anomaly_type_aupr(val_scores, val_labels, *CONTEXTUAL_LEN):.4f}")
    print(f"  Collective {anomaly_type_aupr(val_scores, val_labels, *COLLECTIVE_LEN):.4f}")
    print(f"\n  [Test] 유형별 AUPR")
    print(f"  Point      {anomaly_type_aupr(test_scores, test_labels, *POINT_LEN):.4f}")
    print(f"  Contextual {anomaly_type_aupr(test_scores, test_labels, *CONTEXTUAL_LEN):.4f}")
    print(f"  Collective {anomaly_type_aupr(test_scores, test_labels, *COLLECTIVE_LEN):.4f}")

    # 11. 시각화
    plot_full(val_scores, val_labels, test_scores, test_labels,
              tag="010_lof_diff_x_f8_window_scaler_w50",
              save_path=str(OUTPUT_DIR / "010_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="010_lof_diff_x_f8_window_scaler_w50",
               save_path=str(OUTPUT_DIR / "010_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="010_lof_diff_x_f8_window_scaler_w50",
                    save_path=str(OUTPUT_DIR / "010_score_hist.png"))
