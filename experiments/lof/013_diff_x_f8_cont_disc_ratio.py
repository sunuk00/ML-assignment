"""실험 013: LOF — x_f8 차분 + 연속형 원본 + 이산형 Rolling Mean + StandardScaler

x_f8 채널에만 diff를 적용하고, 연속형 채널(7개)에 StandardScaler를 적용합니다.
이산형 채널(3개)에 Rolling Mean(활성화 비율)을 계산하여 연결합니다.
LocalOutlierFactor(novelty=True)로 학습 및 추론합니다.

피처 차원: 연속형 7(StandardScaler) + 이산형 3(rolling mean) = 10
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
from src.features      import DISCRETE_COLS
from src.models        import fit_lof
from src.ensemble      import flip_score, rank_normalize
from src.evaluate      import evaluate_aupr, evaluate_auroc, anomaly_type_aupr, plot_full, plot_score_hist, plot_zooms

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "lof" / "outputs"

N_NEIGHBORS   = 10
CONTAMINATION = 0.0001
DIFF_COL      = "x_f8"
DISC_WINDOW   = 10
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


def _disc_ratio(df: pd.DataFrame, disc_cols: list[str], window: int) -> np.ndarray:
    return (
        df[disc_cols]
        .rolling(window, min_periods=1)
        .mean()
        .fillna(0)
        .to_numpy()
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 013 LOF — x_f8 Diff + Raw(Cont) + Disc Ratio(W={}) + StandardScaler ===\n".format(DISC_WINDOW))

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

    # 4. 채널 분리
    x_cols    = [c for c in train_df.columns if c.startswith("x_")]
    disc_cols = [c for c in x_cols if c in DISCRETE_COLS]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
    print(f"  연속형: {len(cont_cols)}채널 (x_f8 diff 적용됨)  이산형: {len(disc_cols)}채널")

    # 5. 연속형: StandardScaler (LOF-012 동일)
    scaler       = fit_scaler(train_df[cont_cols])
    train_scaled = apply_scaler(scaler, train_df[cont_cols]).to_numpy()
    val_scaled   = apply_scaler(scaler, val_df[cont_cols]).to_numpy()
    test_scaled  = apply_scaler(scaler, test_df[cont_cols]).to_numpy()

    # 6. 이산형: rolling mean (활성화 비율)
    train_disc = _disc_ratio(train_df, disc_cols, DISC_WINDOW)
    val_disc   = _disc_ratio(val_df,   disc_cols, DISC_WINDOW)
    test_disc  = _disc_ratio(test_df,  disc_cols, DISC_WINDOW)

    # 7. concat: 스케일된 연속형 + 이산형 비율
    train_X = np.hstack([train_scaled, train_disc])
    val_X   = np.hstack([val_scaled,   val_disc])
    test_X  = np.hstack([test_scaled,  test_disc])
    print(f"  feature dim: {train_X.shape[1]}  (cont {len(cont_cols)}ch scaler + disc {len(disc_cols)}ch ratio W={DISC_WINDOW})")

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
              tag="013_lof_diff_x_f8_cont_disc_ratio",
              save_path=str(OUTPUT_DIR / "013_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="013_lof_diff_x_f8_cont_disc_ratio",
               save_path=str(OUTPUT_DIR / "013_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="013_lof_diff_x_f8_cont_disc_ratio",
                    save_path=str(OUTPUT_DIR / "013_score_hist.png"))
