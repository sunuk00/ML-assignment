"""실험 007: IF — 연속형 Rolling 통계 + 이산형 Rolling Mean

연속형 채널(7개)에 Rolling 통계(mean/std/min/max/range)를, 이산형 채널(3개)에 Rolling Mean(활성화 비율)을 사용합니다.
IsolationForest로 학습 및 추론합니다.

피처 차원: 연속형 7 × 5통계 + 이산형 3(rolling mean) = 38
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
from src.features      import rolling_features, DISCRETE_COLS
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


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 007 IF — Rolling cont stats + Discrete window-mean ===\n")

    # 1. 데이터 로드
    train_df, _           = load_data("train",       str(DATA_DIR))
    val_df,   val_labels  = load_data("val",         str(DATA_DIR))
    test_df,  test_labels = load_data("test_public", str(DATA_DIR))
    print(f"  train {train_df.shape}  val {val_df.shape}  test {test_df.shape}")

    # 2. 결측치 처리
    train_df = fill_missing(train_df)
    val_df   = fill_missing(val_df)
    test_df  = fill_missing(test_df)

    # 3. 채널 분리
    x_cols = [c for c in train_df.columns if c.startswith("x_")]
    disc_cols = [c for c in x_cols if c in DISCRETE_COLS]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]

    # 4. 연속형: rolling 통계
    train_cont = rolling_features(train_df, cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)
    val_cont   = rolling_features(val_df,   cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)
    test_cont  = rolling_features(test_df,  cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)

    # 5. 이산형: 같은 윈도우로 mean(활성화 비율) 계산 -> 연속형으로 치환
    train_disc = train_df[disc_cols].rolling(WINDOW_SIZE, min_periods=1).mean().fillna(0)
    val_disc   = val_df[disc_cols].rolling(WINDOW_SIZE, min_periods=1).mean().fillna(0)
    test_disc  = test_df[disc_cols].rolling(WINDOW_SIZE, min_periods=1).mean().fillna(0)

    # 6. concat (열 단위로 통계 피처 + 이산 비율)
    train_X = np.hstack([train_cont.to_numpy(), train_disc[disc_cols].to_numpy()])
    val_X   = np.hstack([val_cont.to_numpy(),   val_disc[disc_cols].to_numpy()])
    test_X  = np.hstack([test_cont.to_numpy(),  test_disc[disc_cols].to_numpy()])
    print(f"  feature dim: {train_X.shape[1]}  (cont {len(cont_cols)}×{len(STATS)} + disc {len(disc_cols)}×ratio)")

    # 7. 모델 학습
    model = fit_isolation_forest(train_X, n_estimators=N_ESTIMATORS,
                                  contamination=CONTAMINATION, random_state=RANDOM_STATE)

    # 8. Score 계산
    val_scores  = rank_normalize(flip_score(model.score_samples(val_X)))
    test_scores = rank_normalize(flip_score(model.score_samples(test_X)))

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
              tag="007_windowed_discrete_mean", save_path=str(OUTPUT_DIR / "007_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="007_windowed_discrete_mean", save_path=str(OUTPUT_DIR / "007_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="007_windowed_discrete_mean", save_path=str(OUTPUT_DIR / "007_score_hist.png"))
