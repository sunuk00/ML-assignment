"""실험 003: IF — 차분 + Rolling 통계

모든 채널(10개)에 diff → Rolling 통계(mean/std/min/max/range)를 적용합니다.
IsolationForest로 학습 및 추론합니다.

피처 차원: 10채널 × 5통계 = 50
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
from src.features      import rolling_features
from src.models        import fit_isolation_forest
from src.ensemble      import flip_score, rank_normalize
from src.evaluate      import evaluate_aupr, evaluate_auroc, anomaly_type_aupr, plot_full, plot_zooms, plot_score_hist

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "isolation_forest" / "outputs"

WINDOW_SIZE   = 50
STATS         = ['mean', 'std', 'min', 'max', 'range']
N_ESTIMATORS  = 300
CONTAMINATION = 0.0001
RANDOM_STATE  = 42
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)


def _apply_diff(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[cols] = df[cols].diff(periods=1).fillna(0)
    return df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 003 IF — Diff + Rolling(w=50) ===\n")

    # 1. 데이터 로드
    train_df, _           = load_data("train",       str(DATA_DIR))
    val_df,   val_labels  = load_data("val",         str(DATA_DIR))
    test_df,  test_labels = load_data("test_public", str(DATA_DIR))
    print(f"  train {train_df.shape}  val {val_df.shape}  test {test_df.shape}")

    # 2. 결측치 처리
    train_df = fill_missing(train_df)
    val_df   = fill_missing(val_df)
    test_df  = fill_missing(test_df)

    # 3. 차분 적용
    x_cols   = [c for c in train_df.columns if c.startswith("x_")]
    train_df = _apply_diff(train_df, x_cols)
    val_df   = _apply_diff(val_df,   x_cols)
    test_df  = _apply_diff(test_df,  x_cols)

    # 4. Rolling 통계 피처 추출 (이산/연속 구분 없이 전체 채널)
    train_X = rolling_features(train_df, cols=x_cols, window_size=WINDOW_SIZE, stats=STATS).to_numpy()
    val_X   = rolling_features(val_df,   cols=x_cols, window_size=WINDOW_SIZE, stats=STATS).to_numpy()
    test_X  = rolling_features(test_df,  cols=x_cols, window_size=WINDOW_SIZE, stats=STATS).to_numpy()
    print(f"  feature dim: {train_X.shape[1]}  ({len(x_cols)} 채널 × {len(STATS)} 통계)")

    # 5. 모델 학습
    model = fit_isolation_forest(train_X, n_estimators=N_ESTIMATORS,
                                  contamination=CONTAMINATION, random_state=RANDOM_STATE)

    # 6. Score 계산
    val_scores  = rank_normalize(flip_score(model.score_samples(val_X)))
    test_scores = rank_normalize(flip_score(model.score_samples(test_X)))

    # 7. 평가
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

    # 8. 시각화
    plot_full(val_scores, val_labels, test_scores, test_labels,
              tag="003_diff_rolling_w50", save_path=str(OUTPUT_DIR / "003_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="003_diff_rolling_w50", save_path=str(OUTPUT_DIR / "003_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="003_diff_rolling_w50", save_path=str(OUTPUT_DIR / "003_score_hist.png"))
