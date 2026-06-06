"""실험 007: Spectral Residual (Saliency Map, 모델 학습 없음)

푸리에 변환 기반의 Spectral Residual로 순간적인 이상 신호를 증폭합니다.
"평범한 주기적 패턴을 제거하고 남은 잔차"를 score로 사용합니다.

알고리즘:
  1. FFT → log 진폭 계산
  2. log 진폭의 rolling mean(w=3) = 평범한 평균 패턴
  3. Spectral Residual = log 진폭 - 평균 패턴
  4. iFFT로 시간 영역 복원 → Saliency Score
  5. 채널별 Saliency 합산 → 최종 score

가설: Rolling 스무딩을 완전히 제거하여 Point Anomaly 신호를 극대화합니다.

결과 (기록):
  val  AUROC=0.5677  AUPR=0.1808
  test AUROC=0.5850  AUPR=0.1809

  [Val] 유형별 AUPR
  Point      0.0546
  Contextual 0.0221
  Collective 0.1645

  [Test] 유형별 AUPR
  Point      0.2092
  Contextual 0.0242
  Collective 0.1631
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader   import load_data
from src.preprocessing import fill_missing
from src.features      import DISCRETE_COLS
from src.ensemble      import rank_normalize
from src.evaluate      import evaluate_aupr, evaluate_auroc, anomaly_type_aupr, plot_full, plot_zooms

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "isolation_forest" / "outputs"

POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)


def _spectral_residual(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """각 채널의 Spectral Residual Saliency Score 합산을 반환합니다."""
    result = pd.DataFrame(index=df.index)
    eps = 1e-8
    for col in cols:
        vals     = df[col].values
        fft_vals = fft(vals)
        mag      = np.abs(fft_vals)
        phase    = np.angle(fft_vals)
        log_mag  = np.log(mag + eps)
        avg_log  = pd.Series(log_mag).rolling(window=3, center=True, min_periods=1).mean().values
        spec_res = log_mag - avg_log
        saliency = np.abs(ifft(np.exp(spec_res + 1j * phase)))
        saliency = pd.Series(saliency).rolling(window=3, center=True, min_periods=1).mean().values
        result[f"{col}_sr"] = saliency
    return result.sum(axis=1).to_numpy()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 007 Spectral Residual Saliency (No Model) ===\n")

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
    print(f"  연속형: {len(cont_cols)}채널")

    # 4. Saliency score 계산
    val_scores  = rank_normalize(_spectral_residual(val_df,  cont_cols))
    test_scores = rank_normalize(_spectral_residual(test_df, cont_cols))

    # 5. 평가
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

    # 6. 시각화
    plot_full(val_scores, val_labels, test_scores, test_labels,
              tag="007_saliency", save_path=str(OUTPUT_DIR / "007_val_score_trace.png"))
    plot_zooms(val_scores, val_labels, test_scores, test_labels,
               tag="007_saliency", save_path=str(OUTPUT_DIR / "007_val_score_zoom.png"))
