"""실험 006: Rolling Z-Score (모델 학습 없음)

IsolationForest 없이 rolling z-score를 anomaly score로 직접 사용합니다.
"현재 값이 최근 window 대비 얼마나 이상한가"를 채널별로 계산 후
최대값을 최종 score로 사용합니다.

가설: Point anomaly처럼 순간적으로 튀는 이상은 모델 학습보다
      z-score 직접 계산이 더 직관적인 신호를 줄 수 있습니다.

결과 (기록):
 val  AUROC=0.5176  AUPR=0.1561
  test AUROC=0.4750  AUPR=0.1278

  [Val] 유형별 AUPR
  Point      0.0019
  Contextual 0.0146
  Collective 0.1462

  [Test] 유형별 AUPR
  Point      0.0019
  Contextual 0.0150
  Collective 0.1158
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader   import load_data
from src.preprocessing import fill_missing
from src.features      import DISCRETE_COLS
from src.ensemble      import rank_normalize
from src.evaluate      import evaluate_aupr, evaluate_auroc, anomaly_type_aupr, plot_full, plot_zooms

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "isolation_forest" / "outputs"

WINDOW_SIZE    = 300   # z-score 계산에 사용할 rolling window
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)


def _rolling_zscore_max(df: pd.DataFrame, cols: list[str], window: int) -> np.ndarray:
    """각 채널의 rolling z-score 절댓값을 계산하고 채널 최대값을 반환합니다."""
    z_abs = pd.DataFrame(index=df.index)
    for col in cols:
        s    = df[col]
        mean = s.rolling(window=window, min_periods=1).mean()
        std  = s.rolling(window=window, min_periods=1).std().fillna(1).replace(0, 1)
        z_abs[col] = ((s - mean) / std).abs()
    return z_abs.max(axis=1).to_numpy()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 006 Rolling Z-Score (No Model) ===\n")

    # 1. 데이터 로드
    train_df, _           = load_data("train",       str(DATA_DIR))
    val_df,   val_labels  = load_data("val",         str(DATA_DIR))
    test_df,  test_labels = load_data("test_public", str(DATA_DIR))
    print(f"  train {train_df.shape}  val {val_df.shape}  test {test_df.shape}")

    # 2. 결측치 처리
    train_df = fill_missing(train_df)
    val_df   = fill_missing(val_df)
    test_df  = fill_missing(test_df)

    # 3. 연속형 채널만 사용 (이산형 z-score는 의미 없음)
    x_cols    = [c for c in train_df.columns if c.startswith("x_")]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
    print(f"  연속형: {len(cont_cols)}채널  window={WINDOW_SIZE}")

    # 4. Rolling z-score score 계산 (각 데이터셋의 자체 rolling 기준)
    val_scores  = rank_normalize(_rolling_zscore_max(val_df,  cont_cols, WINDOW_SIZE))
    test_scores = rank_normalize(_rolling_zscore_max(test_df, cont_cols, WINDOW_SIZE))

    # 5. 평가
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

    # 6. 시각화
    plot_full(val_scores, val_labels, test_scores, test_labels,
              tag="006_zscore", save_path=str(OUTPUT_DIR / "006_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="006_zscore", save_path=str(OUTPUT_DIR / "006_val_score_zoom.png"))
