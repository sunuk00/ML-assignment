"""실험 009: LOF — Rolling 통계 + StandardScaler

연속형 채널(7개)에 Rolling 통계(mean/std/max/min/range) → StandardScaler를 적용합니다.
이산형 채널(3개)은 포함하지 않습니다.
LocalOutlierFactor(novelty=True)로 학습 및 추론합니다.

피처 차원: 연속형 7 × 5통계 → StandardScaler
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
WINDOW_SIZE    = 5
STATS          = ['mean', 'std', 'min', 'max', 'median']
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)

def minmax(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 009 LOF — Rolling Window Stats(Cont, W=50) + StandardScaler ===\n")

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
    x_cols    = [c for c in train_df.columns if c.startswith('x_')]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
    print(f"  연속형: {len(cont_cols)}채널 (이산형 제외)")

    # 4. 윈도우 통계(feature 생성) — train 기준으로 fit할 scaler는 이 DataFrame을 사용
    train_feats = rolling_features(train_df, cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)
    val_feats   = rolling_features(val_df,   cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)
    test_feats  = rolling_features(test_df,  cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)

    # 5. StandardScaler (train 기준)
    scaler = fit_scaler(train_feats)
    train_scaled = apply_scaler(scaler, train_feats)
    val_scaled   = apply_scaler(scaler, val_feats)
    test_scaled  = apply_scaler(scaler, test_feats)

    # 6. 피처 준비
    train_X = train_scaled.to_numpy()
    val_X   = val_scaled.to_numpy()
    test_X  = test_scaled.to_numpy()
    print(f"  feature dim: {train_X.shape[1]}  (window_stats W={WINDOW_SIZE} + scaler)")

    # 7. LOF 학습
    model = fit_lof(train_X, n_neighbors=N_NEIGHBORS, contamination=CONTAMINATION)

    # 8. Score 계산
    # val_scores  = rank_normalize(flip_score(model.score_samples(val_X)))
    # test_scores = rank_normalize(flip_score(model.score_samples(test_X)))
    
    val_scores  = minmax(-model.score_samples(val_X))
    test_scores = minmax(-model.score_samples(test_X))

    # 9. 평가 출력
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
    
    # 10. 시각화
    plot_full(val_scores, val_labels, test_scores, test_labels,
              tag="009_lof_window50_cont_scaler", save_path=str(OUTPUT_DIR / "009_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="009_lof_window50_cont_scaler", save_path=str(OUTPUT_DIR / "009_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="009_lof_window50_cont_scaler", save_path=str(OUTPUT_DIR / "009_score_hist.png"))