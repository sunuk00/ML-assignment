"""실험 013: IF — 연속형 롤링 통계 + 이산형 윈도우 평균(활성화 비율) + x_f8 diff

007 기반에서 x_f8 채널만 1차 차분(diff)을 추가 적용합니다.
연속형 7채널에 대해 rolling(W=100) 통계 5개를 사용하고,
이산형 3채널에 대해서는 rolling mean(비율)을 계산합니다.
x_f8은 장기 트렌드(drift)가 있어 diff()로 비정상성을 제거합니다.

결과 (기록):
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

WINDOW_SIZE   = 100
STATS         = ['mean', 'std', 'min', 'max', 'range']
N_ESTIMATORS  = 300
CONTAMINATION = 0.0001
RANDOM_STATE  = 42
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)



def apply_diff(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """지정 채널을 1차 차분으로 교체합니다 (첫 번째 행은 0으로 채움)."""
    df = df.copy()
    df[col] = df[col].diff().fillna(0)
    return df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 013 IF — Rolling cont stats + Disc Ratio + x_f8 Diff(W=100) ===\n")

    # 1. 데이터 로드
    train_df, _           = load_data("train",       str(DATA_DIR))
    val_df,   val_labels  = load_data("val",         str(DATA_DIR))
    test_df,  test_labels = load_data("test_public", str(DATA_DIR))
    print(f"  train {train_df.shape}  val {val_df.shape}  test {test_df.shape}")

    # 2. 결측치 처리
    train_df = fill_missing(train_df)
    val_df   = fill_missing(val_df)
    test_df  = fill_missing(test_df)

    # 3. x_f8 diff 적용
    train_df = apply_diff(train_df, 'x_f8')
    val_df   = apply_diff(val_df,   'x_f8')
    test_df  = apply_diff(test_df,  'x_f8')

    # 4. 채널 분리
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
              tag="013_diff_x_f8_ratio_rolling_w100", save_path=str(OUTPUT_DIR / "013_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="013_diff_x_f8_ratio_rolling_w100", save_path=str(OUTPUT_DIR / "013_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="013_diff_x_f8_ratio_rolling_w100", save_path=str(OUTPUT_DIR / "013_score_hist.png"))
