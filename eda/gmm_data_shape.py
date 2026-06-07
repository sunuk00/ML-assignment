"""EDA: GMM 적합성 진단 — PCA 공간에서 데이터 분포 형태 확인

GMM 학습 전, 데이터가 타원형 가우시안 군집인지 복잡한 다봉 분포인지 시각적으로 확인합니다.
GMM 실험(001~003)과 동일한 전처리 파이프라인 사용:
    연속형 7채널 → StandardScaler → PCA(95%) → 시각화

생성 파일:
  outputs/gmm_shape_01_scatter.png  — PC 쌍별 산점도 + KDE 등고선
  outputs/gmm_shape_02_marginal.png — 각 PC 성분의 주변 분포 + 정규분포 피팅
  outputs/gmm_shape_03_scree.png    — PCA 설명 분산 비율 (스크리 플롯)
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.stats import norm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader   import load_data
from src.preprocessing import fill_missing, fit_scaler, apply_scaler, fit_pca, apply_pca
from src.features      import DISCRETE_COLS

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PCA_VARIANCE = 0.95


def _scatter_pair(ax, train_pc, val_normal_pc, val_anomaly_pc, i, j):
    """PC i vs PC j 산점도 + train KDE 등고선."""
    ax.scatter(train_pc[::10, i], train_pc[::10, j],
               c="lightgray", s=3, alpha=0.4, label="Train (정상)")
    ax.scatter(val_normal_pc[::20, i], val_normal_pc[::20, j],
               c="steelblue", s=4, alpha=0.3, label="Val 정상")
    ax.scatter(val_anomaly_pc[:, i], val_anomaly_pc[:, j],
               c="tomato", s=8, alpha=0.7, label="Val 이상")

    # KDE 등고선 — train 분포가 타원형인지 확인
    try:
        sample = train_pc[::20, :]
        kde = stats.gaussian_kde(sample[:, [i, j]].T)
        xi = np.linspace(train_pc[:, i].min(), train_pc[:, i].max(), 70)
        yi = np.linspace(train_pc[:, j].min(), train_pc[:, j].max(), 70)
        XI, YI = np.meshgrid(xi, yi)
        ZI = kde(np.vstack([XI.ravel(), YI.ravel()])).reshape(XI.shape)
        ax.contour(XI, YI, ZI, levels=6, colors="navy", alpha=0.5, linewidths=0.8)
    except Exception:
        pass

    ax.set_xlabel(f"PC{i + 1}")
    ax.set_ylabel(f"PC{j + 1}")
    ax.set_title(f"PC{i + 1} vs PC{j + 1}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== GMM 적합성 진단 ===\n")

    # 1. 데이터 로드
    train_df, _            = load_data("train",       str(DATA_DIR))
    val_df,   val_labels   = load_data("val",         str(DATA_DIR))
    print(f"  train {train_df.shape}  val {val_df.shape}")

    # 2. 결측치 처리
    train_df = fill_missing(train_df)
    val_df   = fill_missing(val_df)

    # 3. 연속형 채널만 사용
    x_cols    = [c for c in train_df.columns if c.startswith("x_")]
    cont_cols = [c for c in x_cols if c not in DISCRETE_COLS]
    print(f"  연속형: {len(cont_cols)}채널")

    # 4. StandardScaler + PCA (GMM 실험과 동일)
    scaler       = fit_scaler(train_df[cont_cols])
    train_scaled = apply_scaler(scaler, train_df[cont_cols])
    val_scaled   = apply_scaler(scaler, val_df[cont_cols])

    pca       = fit_pca(train_scaled, variance=PCA_VARIANCE)
    train_pca = apply_pca(pca, train_scaled).to_numpy()
    val_pca   = apply_pca(pca, val_scaled).to_numpy()

    n_comp = train_pca.shape[1]
    val_normal  = val_pca[val_labels == 0]
    val_anomaly = val_pca[val_labels == 1]
    print(f"  PCA components: {n_comp}  (variance={PCA_VARIANCE:.2f})")
    print(f"  val 정상: {len(val_normal):,}  이상: {len(val_anomaly):,}\n")

    # ── 플롯 1: PC 쌍별 산점도 + KDE 등고선 ──────────────────────
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("PC 쌍별 산점도 — 군집 형태 확인 (등고선=Train 분포)", fontsize=12)

    for ax, (i, j) in zip(axes, pairs):
        _scatter_pair(ax, train_pca, val_normal, val_anomaly, i, j)

    axes[0].legend(fontsize=8, markerscale=3)
    plt.tight_layout()
    out1 = OUTPUT_DIR / "gmm_shape_01_scatter.png"
    plt.savefig(str(out1), dpi=120)
    plt.close()
    print(f"  저장: {out1}")

    # ── 플롯 2: 각 PC 성분의 주변 분포 + 정규분포 피팅 ──────────
    ncols = min(4, n_comp)
    nrows = (n_comp + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes_flat = np.array(axes).flatten()
    fig.suptitle("각 PC 성분의 주변 분포 (Train) — 정규분포 피팅 비교", fontsize=12)

    for k in range(n_comp):
        ax   = axes_flat[k]
        data = train_pca[:, k]
        ax.hist(data, bins=80, density=True, alpha=0.6, color="steelblue", label="Train")

        mu, sigma = norm.fit(data)
        x_range   = np.linspace(data.min(), data.max(), 300)
        ax.plot(x_range, norm.pdf(x_range, mu, sigma),
                "r-", linewidth=1.5, label=f"N(μ={mu:.2f}, σ={sigma:.2f})")

        ax.set_title(f"PC{k + 1}")
        ax.set_xlabel("값")
        ax.set_ylabel("밀도")
        ax.legend(fontsize=7)

    for k in range(n_comp, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout()
    out2 = OUTPUT_DIR / "gmm_shape_02_marginal.png"
    plt.savefig(str(out2), dpi=120)
    plt.close()
    print(f"  저장: {out2}")

    # ── 플롯 3: 스크리 플롯 ──────────────────────────────────────
    explained   = pca.explained_variance_ratio_
    cumulative  = np.cumsum(explained)
    comp_labels = [f"PC{i + 1}" for i in range(n_comp)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("PCA 설명 분산 비율", fontsize=12)

    ax = axes[0]
    ax.bar(comp_labels, explained, color="steelblue", alpha=0.75)
    ax.set_xlabel("주성분")
    ax.set_ylabel("설명 분산 비율")
    ax.set_title("개별 설명 분산 (스크리 플롯)")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1]
    ax.plot(comp_labels, cumulative, "o-", color="steelblue", markersize=5)
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label="95%")
    ax.fill_between(range(n_comp), cumulative, alpha=0.15, color="steelblue")
    ax.set_xlabel("주성분 수")
    ax.set_ylabel("누적 설명 분산 비율")
    ax.set_title("누적 설명 분산")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out3 = OUTPUT_DIR / "gmm_shape_03_scree.png"
    plt.savefig(str(out3), dpi=120)
    plt.close()
    print(f"  저장: {out3}")
