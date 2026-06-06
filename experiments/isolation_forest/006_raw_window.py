"""실험 009: IF — Raw Window Flatten

통계량을 구하지 않고, window 크기만큼의 원본 데이터를 그대로 이어붙여 피처로 사용합니다.
피처 차원: 10채널 × window_size = 5000

결과 (기록):
  val  AUROC=0.6401  AUPR=0.2293
  test AUROC=0.6066  AUPR=0.1750

  [Val] 유형별 AUPR
  Point      0.0004
  Contextual 0.0521
  Collective 0.2056

  [Test] 유형별 AUPR
  Point      0.0007
  Contextual 0.0251
  Collective 0.1575
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
from src.models        import fit_isolation_forest
from src.ensemble      import flip_score, rank_normalize
from src.evaluate      import evaluate_aupr, evaluate_auroc, anomaly_type_aupr, plot_full, plot_zooms, plot_score_hist

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "isolation_forest" / "outputs"

WINDOW_SIZE   = 50
N_ESTIMATORS  = 300
CONTAMINATION = 0.0001
RANDOM_STATE  = 42
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)


def _raw_window_features(df: pd.DataFrame, cols: list[str], window: int) -> np.ndarray:
    """과거 window 길이의 원본 데이터를 flatten하여 피처 행렬로 반환합니다."""
    arr = df[cols].to_numpy()
    N, C = arr.shape
    padded = np.vstack([np.zeros((window - 1, C)), arr])
    result = np.zeros((N, window * C))
    for i in range(N):
        result[i] = padded[i : i + window].flatten()
    return result


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 006 IF — Raw Window Flatten ===\n")

    # 1. 데이터 로드
    train_df, _           = load_data("train",       str(DATA_DIR))
    val_df,   val_labels  = load_data("val",         str(DATA_DIR))
    test_df,  test_labels = load_data("test_public", str(DATA_DIR))
    print(f"  train {train_df.shape}  val {val_df.shape}  test {test_df.shape}")

    # 2. 결측치 처리
    train_df = fill_missing(train_df)
    val_df   = fill_missing(val_df)
    test_df  = fill_missing(test_df)

    # 3. Raw Window 피처 추출
    x_cols  = [c for c in train_df.columns if c.startswith("x_")]
    train_X = _raw_window_features(train_df, x_cols, WINDOW_SIZE)
    val_X   = _raw_window_features(val_df,   x_cols, WINDOW_SIZE)
    test_X  = _raw_window_features(test_df,  x_cols, WINDOW_SIZE)
    print(f"  feature dim: {train_X.shape[1]}  ({len(x_cols)} 채널 × {WINDOW_SIZE} window)")

    # 4. 모델 학습
    model = fit_isolation_forest(train_X, n_estimators=N_ESTIMATORS,
                                  contamination=CONTAMINATION, random_state=RANDOM_STATE)

    # 5. Score 계산
    val_scores  = rank_normalize(flip_score(model.score_samples(val_X)))
    test_scores = rank_normalize(flip_score(model.score_samples(test_X)))

    # 6. 평가
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

    # 7. 시각화
    plot_full(val_scores, val_labels, test_scores, test_labels,
              tag="006_raw_window", save_path=str(OUTPUT_DIR / "006_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="006_raw_window", save_path=str(OUTPUT_DIR / "006_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="006_raw_window", save_path=str(OUTPUT_DIR / "006_score_hist.png"))
