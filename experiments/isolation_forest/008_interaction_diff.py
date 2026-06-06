"""실험 008: IF — 004(rolling separated) 피처 + 채널 쌍 간의 차이(Interaction) 파생 피처 추가
상관관계가 매우 높은 채널 쌍의 차이값을 파생변수로 추가하여 '관계 붕괴'를 직접 포착합니다.

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

WINDOW_SIZE   = 50
STATS         = ['mean', 'std', 'min', 'max', 'range']
N_ESTIMATORS  = 300
CONTAMINATION = 0.0001
RANDOM_STATE  = 42
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)

# EDA에서 관찰된 동기화 쌍 예시 — 필요하면 더 추가
PAIRS = [('x_3a', 'x_06'), ('x_d4', 'x_4b')]


def _add_interaction_diffs(df: pd.DataFrame, pairs: list[tuple[str, str]], window: int) -> pd.DataFrame:
    """각 쌍에 대해 (a - b) 원본과 그 rolling mean/std를 생성하여 반환합니다.
    반환 DataFrame은 시계열 길이와 동일한 index를 가집니다.
    """
    out = pd.DataFrame(index=df.index)
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            diff = df[a] - df[b]
            out[f"diff_{a}_{b}"] = diff
            # rolling summary of the diff to make it more robust
            out[f"diff_{a}_{b}_mean"] = diff.rolling(window, min_periods=1).mean().fillna(0)
            out[f"diff_{a}_{b}_std"] = diff.rolling(window, min_periods=1).std().fillna(0)
    return out


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 008 IF — Rolling separated + Interaction diff features ===\n")

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

    # 4. 기본 004 스타일: 연속형 rolling 통계 + 이산형은 원본(004 방식 재현)
    train_cont = rolling_features(train_df, cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)
    val_cont   = rolling_features(val_df,   cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)
    test_cont  = rolling_features(test_df,  cols=cont_cols, window_size=WINDOW_SIZE, stats=STATS)

    train_disc_raw = train_df[disc_cols].fillna(0)
    val_disc_raw   = val_df[disc_cols].fillna(0)
    test_disc_raw  = test_df[disc_cols].fillna(0)

    # 5. Interaction diff 파생변수 추가
    train_inter = _add_interaction_diffs(train_df, PAIRS, WINDOW_SIZE)
    val_inter   = _add_interaction_diffs(val_df,   PAIRS, WINDOW_SIZE)
    test_inter  = _add_interaction_diffs(test_df,  PAIRS, WINDOW_SIZE)

    # 6. 최종 피처 결합: rolling cont | raw disc | interaction diffs
    train_X = np.hstack([train_cont.to_numpy(), train_disc_raw.to_numpy(), train_inter.to_numpy()])
    val_X   = np.hstack([val_cont.to_numpy(),   val_disc_raw.to_numpy(),   val_inter.to_numpy()])
    test_X  = np.hstack([test_cont.to_numpy(),  test_disc_raw.to_numpy(),  test_inter.to_numpy()])
    print(f"  feature dim: {train_X.shape[1]}  (cont {train_cont.shape[1]} + disc {len(disc_cols)} + inter {train_inter.shape[1]})")

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
              tag="008_interaction_diff", save_path=str(OUTPUT_DIR / "008_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="008_interaction_diff", save_path=str(OUTPUT_DIR / "008_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="008_interaction_diff", save_path=str(OUTPUT_DIR / "008_score_hist.png"))
