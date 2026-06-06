"""실험 002: GMM — Rolling 통계 도입 (Contextual 타겟, W=200)

전처리: 연속형 7채널 → StandardScaler → PCA(95%) → Rolling(W=200) → 5통계량
GMM(n_components=3, covariance_type='full')으로 확률 밀도를 추정합니다.

설계 근거:
Contextual Anomaly 길이(60~90 step)를 충분히 덮기 위해 W=200을 적용합니다.
과거 200 step의 평균/분산 변화를 GMM이 학습하여 정상 분포 궤도에서 벗어나는
이상을 탐지할 수 있는지 확인합니다.
001_baseline 대비 Contextual/Collective AUPR 향상을 기대합니다.

피처 차원: 연속형 7 → Scaler → PCA(95%) → Rolling W=200 → 각 PC × 5통계
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader   import load_data
from src.preprocessing import fill_missing, fit_scaler, apply_scaler, fit_pca, apply_pca
from src.features      import DISCRETE_COLS, rolling_features
from src.models        import fit_gmm
from src.ensemble      import flip_score, rank_normalize
from src.evaluate      import evaluate_aupr, evaluate_auroc, anomaly_type_aupr, plot_full, plot_zooms, plot_score_hist

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "gmm" / "outputs"

WINDOW_SIZE     = 200
ROLL_STATS      = ['mean', 'std', 'min', 'max', 'range']
N_COMPONENTS    = 3
COVARIANCE_TYPE = "full"
PCA_VARIANCE    = 0.95
POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 002 GMM — Rolling(W=200) + Scaler + PCA ===\n")

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
    x_cols    = [c for c in train_df.columns if c.startswith("x_")]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
    print(f"  연속형: {len(cont_cols)}채널 (이산형 제외)")

    # 4. StandardScaler (학습 데이터 기준)
    scaler       = fit_scaler(train_df[cont_cols])
    train_scaled = apply_scaler(scaler, train_df[cont_cols])
    val_scaled   = apply_scaler(scaler, val_df[cont_cols])
    test_scaled  = apply_scaler(scaler, test_df[cont_cols])

    # 5. PCA (학습 데이터 기준)
    pca       = fit_pca(train_scaled, variance=PCA_VARIANCE)
    train_pca = apply_pca(pca, train_scaled)
    val_pca   = apply_pca(pca, val_scaled)
    test_pca  = apply_pca(pca, test_scaled)
    print(f"  PCA components: {train_pca.shape[1]}  (variance={PCA_VARIANCE:.2f})")

    # 6. Rolling 통계 (PCA 성분 기준, W=200)
    pc_cols   = list(train_pca.columns)
    train_X   = rolling_features(train_pca, cols=pc_cols, window_size=WINDOW_SIZE, stats=ROLL_STATS).to_numpy()
    val_X     = rolling_features(val_pca,   cols=pc_cols, window_size=WINDOW_SIZE, stats=ROLL_STATS).to_numpy()
    test_X    = rolling_features(test_pca,  cols=pc_cols, window_size=WINDOW_SIZE, stats=ROLL_STATS).to_numpy()
    print(f"  feature dim: {train_X.shape[1]}  ({len(pc_cols)} PC × {len(ROLL_STATS)} 통계, W={WINDOW_SIZE})")

    # 7. GMM 학습
    print(f"  GMM n_components={N_COMPONENTS}  covariance_type={COVARIANCE_TYPE!r}")
    model = fit_gmm(train_X, n_components=N_COMPONENTS, covariance_type=COVARIANCE_TYPE)

    # 8. Score 계산 (log-likelihood → 작을수록 이상 → flip)
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
              tag="002_gmm_rolling_w200", save_path=str(OUTPUT_DIR / "002_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="002_gmm_rolling_w200", save_path=str(OUTPUT_DIR / "002_val_score_zoom.png"))
    plot_score_hist(val_scores, val_labels, test_scores, test_labels,
                    tag="002_gmm_rolling_w200", save_path=str(OUTPUT_DIR / "002_score_hist.png"))
