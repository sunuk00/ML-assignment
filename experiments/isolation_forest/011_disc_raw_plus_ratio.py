"""실험 011: IF — 연속형 Rolling 통계 + 이산형 원본 상태값 + 이산형 Rolling Mean

전처리: 결측치 처리 후 연속형/이산형 분리, 별도 피처 구성
Feature:
  연속형 7채널 → rolling(W=50) 통계 5개 (mean, std, min, max, range)
  이산형 3채널 → 원본 상태값 (OHE 없이 정수 그대로)
              + rolling mean(W=50) 활성화 비율

설계 근거:
007은 이산형을 rolling mean(비율)만으로 처리합니다.
비율은 '최근 W 구간 내 평균 활성화 빈도'를 표현하지만
현재 시점의 실제 상태(0 vs 1 등 정수값)는 담지 못합니다.

원본 상태값을 함께 넣으면 트리가 아래 분기 조건을 직접 학습합니다:
  raw=1 & ratio≈0  → 오랜 비활성 후 갑작스러운 활성화 (Point 이상 후보)
  raw=0 & ratio≈1  → 오랜 활성 구간에서 갑작스러운 비활성화
  raw=1 & ratio≈1  → 안정적인 활성 구간 (정상)

OHE는 이산 상태 하나를 여러 열로 확장해 트리 분기를 분산시키고
"현재 상태 vs 과거 비율" 간 상호작용을 드러내지 못합니다.
정수 원본 유지 시 트리가 단일 임계값(≈0.5)으로 상태를 구분하므로
rolling mean과의 조합 패턴을 더 효율적으로 학습합니다.

비교 실험:
  007 — 연속형 rolling 통계 + 이산형 rolling mean만 (raw 없음)
  011 — 연속형 rolling 통계 + 이산형 raw + 이산형 rolling mean (본 실험)

피처 차원:
  연속형 7 × 5(rolling 통계) = 35
  이산형 3 × 1(raw)          =  3
  이산형 3 × 1(ratio)        =  3
  합계                        = 41
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
