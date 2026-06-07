"""실험 010: LOF — x_f8 Diff + Z-Score Rolling(W=200) + 이산형 비율

전처리: x_f8 채널만 diff(1).fillna(0) 적용, 나머지 9채널은 원본 유지
Feature: 연속형 7채널 → z-score(W=200), 이산형 3채널 → rolling mean(비율)

설계 근거:
기존 실험들은 rolling mean/std 절대값을 피처로 썼으나,
절대값은 센서마다 스케일이 달라 LOF의 거리 계산이 특정 센서에 쏠릴 수 있습니다.
z-score = (현재값 − 과거 rolling mean) / (과거 rolling std + ε) 는
"과거 맥락 대비 현재가 얼마나 벗어났는가"를 스케일 불변 방식으로 표현하므로
거리 기반 알고리즘인 LOF에 특히 유리할 것으로 기대합니다.
shift(1) 로 현재 시점을 계산에서 제외해 look-ahead bias를 방지합니다.
x_f8은 장기 트렌드가 강한 채널이므로 diff()로 비정상성을 제거합니다.

피처 차원: 연속형 7 × 1(z-score) + 이산형 3 × 1(ratio) = 10
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
from src.models        import fit_lof
from src.ensemble      import flip_score, rank_normalize
from src.evaluate      import evaluate_aupr, evaluate_auroc, anomaly_type_aupr, plot_full, plot_zooms, plot_score_hist

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "lof" / "outputs"

WINDOW_SIZE   = 200
DIFF_COL      = 'x_f8'
N_NEIGHBORS   = 10
CONTAMINATION = 0.0001
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)


def _apply_diff_single(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].diff(periods=1).fillna(0)
    return df


def _zscore_features(df: pd.DataFrame, cols: list[str], w: int) -> pd.DataFrame:
    """각 채널에 대해 z-score = (현재 - 과거 rolling mean) / (과거 rolling std + 1e-8) 계산.
    shift(1)로 현재 시점을 rolling 계산에서 제외해 look-ahead bias를 방지합니다."""
    past   = df[cols].shift(1)
    r_mean = past.rolling(w, min_periods=1).mean()
    r_std  = past.rolling(w, min_periods=1).std().fillna(0)
    z      = (df[cols] - r_mean) / (r_std + 1e-8)
    z.columns = [f"{c}_z" for c in cols]
    return z.fillna(0)


def _disc_ratio(df: pd.DataFrame, cols: list[str], w: int) -> pd.DataFrame:
    return df[cols].rolling(w, min_periods=1).mean().fillna(0)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 010 LOF — x_f8 Diff + Z-Score Rolling(W=200) + Disc Ratio ===\n")

    # 1. 데이터 로드
    train_df, _           = load_data("train",       str(DATA_DIR))
    val_df,   val_labels  = load_data("val",         str(DATA_DIR))
    test_df,  test_labels = load_data("test_public", str(DATA_DIR))
    print(f"  train {train_df.shape}  val {val_df.shape}  test {test_df.shape}")

    # 2. 결측치 처리
    train_df = fill_missing(train_df)
    val_df   = fill_missing(val_df)
    test_df  = fill_missing(test_df)

    # 3. x_f8 채널만 diff 적용
    train_df = _apply_diff_single(train_df, DIFF_COL)
    val_df   = _apply_diff_single(val_df,   DIFF_COL)
    test_df  = _apply_diff_single(test_df,  DIFF_COL)
    print(f"  전처리: {DIFF_COL} → diff(1).fillna(0), 나머지 9채널 원본 유지")

    # 4. 채널 분리
    x_cols    = [c for c in train_df.columns if c.startswith("x_")]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
    disc_cols = [c for c in x_cols if c in DISCRETE_COLS]
    print(f"  연속형: {len(cont_cols)}채널  이산형: {len(disc_cols)}채널")

    # 5. 연속형: z-score 피처 (W=200, look-ahead 없음)
    train_z = _zscore_features(train_df, cont_cols, WINDOW_SIZE)
    val_z   = _zscore_features(val_df,   cont_cols, WINDOW_SIZE)
    test_z  = _zscore_features(test_df,  cont_cols, WINDOW_SIZE)

    # 6. 이산형: rolling mean (활성화 비율)
    train_d = _disc_ratio(train_df, disc_cols, WINDOW_SIZE)
    val_d   = _disc_ratio(val_df,   disc_cols, WINDOW_SIZE)
    test_d  = _disc_ratio(test_df,  disc_cols, WINDOW_SIZE)

    # 7. concat
    train_X = np.hstack([train_z.to_numpy(), train_d.to_numpy()])
    val_X   = np.hstack([val_z.to_numpy(),   val_d.to_numpy()])
    test_X  = np.hstack([test_z.to_numpy(),  test_d.to_numpy()])
    print(f"  feature dim: {train_X.shape[1]}  (cont {len(cont_cols)}×z-score + disc {len(disc_cols)}×ratio, W={WINDOW_SIZE})")

    # 8. 모델 학습
    model = fit_lof(train_X, n_neighbors=N_NEIGHBORS, contamination=CONTAMINATION)

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
              tag="010_lof_zscore_rolling_w200", save_path=str(OUTPUT_DIR / "010_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="010_lof_zscore_rolling_w200", save_path=str(OUTPUT_DIR / "010_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="010_lof_zscore_rolling_w200", save_path=str(OUTPUT_DIR / "010_score_hist.png"))