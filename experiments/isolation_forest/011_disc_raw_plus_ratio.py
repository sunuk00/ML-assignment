"""실험 011: IF — 연속형 Rolling 통계 + 이산형 원본값 + 이산형 Rolling Mean

연속형 채널(7개)에 Rolling 통계(mean/std/min/max/range)를, 이산형 채널(3개)에 원본값 + Rolling Mean(활성화 비율)을 사용합니다.
IsolationForest로 학습 및 추론합니다.

피처 차원: 연속형 7 × 5통계 + 이산형 3(raw + rolling mean) = 41
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
from src.features      import DISCRETE_COLS, rolling_features
from src.models        import fit_isolation_forest
from src.ensemble      import flip_score, rank_normalize
from src.evaluate      import evaluate_aupr, evaluate_auroc, anomaly_type_aupr, plot_full, plot_zooms, plot_score_hist

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "isolation_forest" / "outputs"

WINDOW_SIZE   = 50
STATS         = ["mean", "std", "min", "max", "range"]
N_ESTIMATORS  = 300
CONTAMINATION = 0.0001
RANDOM_STATE  = 42
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)


def _disc_ratio(df: pd.DataFrame, cols: list[str], w: int) -> pd.DataFrame:
    """이산형 sliding window 내 활성화 비율 (rolling mean)."""
    ratio = df[cols].rolling(w, min_periods=1).mean().fillna(0)
    ratio.columns = [f"{c}_ratio" for c in cols]
    return ratio


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 011 IF — Cont Rolling Stats + Disc Raw + Disc Ratio(W=50) ===\n")

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
    x_cols    = [c for c in train_df.columns if c.startswith("x_")]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
    disc_cols = [c for c in x_cols if c in DISCRETE_COLS]
    print(f"  연속형: {len(cont_cols)}채널  이산형: {len(disc_cols)}채널")

    # 4. 연속형: rolling 통계 5개 (007과 동일)
    train_cont = rolling_features(train_df, cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)
    val_cont   = rolling_features(val_df,   cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)
    test_cont  = rolling_features(test_df,  cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)

    # 5. 이산형: 원본 상태값 (OHE 없이 정수 그대로)
    train_dr = train_df[disc_cols].copy()
    val_dr   = val_df[disc_cols].copy()
    test_dr  = test_df[disc_cols].copy()

    # 6. 이산형: sliding window 내 활성화 비율 (rolling mean)
    train_dm = _disc_ratio(train_df, disc_cols, WINDOW_SIZE)
    val_dm   = _disc_ratio(val_df,   disc_cols, WINDOW_SIZE)
    test_dm  = _disc_ratio(test_df,  disc_cols, WINDOW_SIZE)

    # 7. concat: cont rolling stats | disc raw | disc ratio
    train_X = np.hstack([train_cont.to_numpy(), train_dr.to_numpy(), train_dm.to_numpy()])
    val_X   = np.hstack([val_cont.to_numpy(),   val_dr.to_numpy(),   val_dm.to_numpy()])
    test_X  = np.hstack([test_cont.to_numpy(),  test_dr.to_numpy(),  test_dm.to_numpy()])
    print(f"  feature dim: {train_X.shape[1]}"
          f"  (cont {len(cont_cols)}×{len(STATS)}stats + disc {len(disc_cols)}×raw + disc {len(disc_cols)}×ratio, W={WINDOW_SIZE})")

    # 8. 모델 학습
    print(f"  IF n_estimators={N_ESTIMATORS}  contamination={CONTAMINATION}")
    model = fit_isolation_forest(train_X, n_estimators=N_ESTIMATORS,
                                  contamination=CONTAMINATION, random_state=RANDOM_STATE)

    # 9. Score 계산
    val_scores  = rank_normalize(flip_score(model.score_samples(val_X)))
    test_scores = rank_normalize(flip_score(model.score_samples(test_X)))

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
              tag="011_disc_raw_plus_ratio",
              save_path=str(OUTPUT_DIR / "011_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="011_disc_raw_plus_ratio",
               save_path=str(OUTPUT_DIR / "011_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="011_disc_raw_plus_ratio",
                    save_path=str(OUTPUT_DIR / "011_score_hist.png"))
