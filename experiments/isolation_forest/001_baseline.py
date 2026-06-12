"""실험 001: IF — 원본 피처 (Raw Features)

전처리 없이 원본 x_ 채널(10개)을 그대로 Isolation Forest에 입력합니다.
baseline 성능의 하한선을 확인하는 실험입니다.

결과 (기록):
  val  AUROC=0.7164  AUPR=0.3298
  test AUROC=0.6360  AUPR=0.2201

  [Val] 유형별 AUPR
  Point      0.0031
  Contextual 0.1067
  Collective 0.2883

  [Test] 유형별 AUPR
  Point      0.0175
  Contextual 0.0601
  Collective 0.1893
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

N_ESTIMATORS  = 100
CONTAMINATION = 0.00001
RANDOM_STATE  = 42
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 001 IF — Raw Features ===\n")

    # 1. 데이터 로드
    train_df, _           = load_data("train",       str(DATA_DIR))
    val_df,   val_labels  = load_data("val",         str(DATA_DIR))
    test_df,  test_labels = load_data("test_public", str(DATA_DIR))
    print(f"  train {train_df.shape}  val {val_df.shape}  test {test_df.shape}")

    # 2. 결측치 처리
    train_df = fill_missing(train_df)
    val_df   = fill_missing(val_df)
    test_df  = fill_missing(test_df)

    # 3. 피처: 원본 x_ 채널 그대로
    x_cols  = [c for c in train_df.columns if c.startswith("x_")]
    train_X = train_df[x_cols].to_numpy()
    val_X   = val_df[x_cols].to_numpy()
    test_X  = test_df[x_cols].to_numpy()
    print(f"  feature dim: {train_X.shape[1]}  ({len(x_cols)} 채널 원본)")

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
              tag="001_baseline", save_path=str(OUTPUT_DIR / "001_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="001_baseline", save_path=str(OUTPUT_DIR / "001_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="001_baseline", save_path=str(OUTPUT_DIR / "001_score_hist.png"))
