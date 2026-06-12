"""앙상블 009: LOF-013 + GMM-005 — Rank 평균 앙상블

LOF: x_f8 차분 → 연속형 7채널 원본 + 이산형 Rolling Mean → StandardScaler (LOF-013 기반)
GMM: x_f8 차분 → 연속형 7채널 → StandardScaler → PCA → Rolling 통계 (GMM-005 기반)
rank_normalize 후 단순 평균으로 앙상블합니다.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, gaussian_kde

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from sklearn.metrics import roc_curve, precision_recall_curve

from src.data_loader   import load_data
from src.preprocessing import fill_missing, fit_scaler, apply_scaler, fit_pca, apply_pca
from src.features      import DISCRETE_COLS, rolling_features
from src.models        import fit_lof, fit_gmm
from src.ensemble      import flip_score, rank_normalize, ensemble_scores
from src.evaluate      import (evaluate_aupr, evaluate_auroc, anomaly_type_aupr,
                               plot_full, plot_zooms, plot_score_hist, _draw_full,
                               _find_segments)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "ensemble" / "outputs"

# ── LOF 하이퍼파라미터 (LOF-013 기반) ───────────────────────────
LOF_NEIGHBORS  = 10
LOF_CONTAM     = 0.0001
LOF_DIFF_COL   = "x_f8"
LOF_DISC_WINDOW = 10

# ── GMM 하이퍼파라미터 (GMM-005 기반) ───────────────────────────
GMM_DIFF_COL    = "x_f8"
GMM_WINDOW      = 20
GMM_STATS       = ["mean", "std", "min", "max", "range"]
GMM_N_COMP      = 5
GMM_COV_TYPE    = "full"
GMM_PCA_VAR     = 0.95

POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)

VIZ_MINMAX = True


def minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def _apply_diff(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].diff().fillna(0)
    return df


# ── LOF 피처 빌더 (LOF-013) ──────────────────────────────────────

def build_lof_features(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    test_df:   pd.DataFrame,
    cont_cols: list[str],
    disc_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr_d = _apply_diff(train_df, LOF_DIFF_COL)
    va_d = _apply_diff(val_df,   LOF_DIFF_COL)
    te_d = _apply_diff(test_df,  LOF_DIFF_COL)

    scaler = fit_scaler(tr_d[cont_cols])
    tr_scaled = apply_scaler(scaler, tr_d[cont_cols]).to_numpy()
    va_scaled = apply_scaler(scaler, va_d[cont_cols]).to_numpy()
    te_scaled = apply_scaler(scaler, te_d[cont_cols]).to_numpy()

    def _disc_ratio(df: pd.DataFrame) -> np.ndarray:
        return df[disc_cols].rolling(LOF_DISC_WINDOW, min_periods=1).mean().fillna(0).to_numpy()

    return (
        np.hstack([tr_scaled, _disc_ratio(tr_d)]),
        np.hstack([va_scaled, _disc_ratio(va_d)]),
        np.hstack([te_scaled, _disc_ratio(te_d)]),
    )


# ── GMM 피처 빌더 (GMM-005) ──────────────────────────────────────

def build_gmm_features(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    test_df:   pd.DataFrame,
    cont_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    tr_d = _apply_diff(train_df, GMM_DIFF_COL)
    va_d = _apply_diff(val_df,   GMM_DIFF_COL)
    te_d = _apply_diff(test_df,  GMM_DIFF_COL)

    scaler    = fit_scaler(tr_d[cont_cols])
    tr_scaled = apply_scaler(scaler, tr_d[cont_cols])
    va_scaled = apply_scaler(scaler, va_d[cont_cols])
    te_scaled = apply_scaler(scaler, te_d[cont_cols])

    pca    = fit_pca(tr_scaled, variance=GMM_PCA_VAR)
    tr_pca = apply_pca(pca, tr_scaled)
    va_pca = apply_pca(pca, va_scaled)
    te_pca = apply_pca(pca, te_scaled)

    pc_cols = list(tr_pca.columns)
    tr_X = rolling_features(tr_pca, cols=pc_cols, window_size=GMM_WINDOW, stats=GMM_STATS).to_numpy()
    va_X = rolling_features(va_pca, cols=pc_cols, window_size=GMM_WINDOW, stats=GMM_STATS).to_numpy()
    te_X = rolling_features(te_pca, cols=pc_cols, window_size=GMM_WINDOW, stats=GMM_STATS).to_numpy()
    return tr_X, va_X, te_X, pca


# ── 다양성 분석 시각화 ───────────────────────────────────────────

def plot_diversity(
    lof_val:     np.ndarray,
    gmm_val:     np.ndarray,
    val_labels:  np.ndarray,
    lof_test:    np.ndarray,
    gmm_test:    np.ndarray,
    test_labels: np.ndarray,
    ens_val:     np.ndarray,
    save_path:   str,
    viz_lof_val: np.ndarray | None = None,
    viz_gmm_val: np.ndarray | None = None,
    viz_ens_val: np.ndarray | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("앙상블 009 — 모델 다양성 진단 (LOF-013 × GMM-005)", fontsize=13)

    pcc_val,  _ = pearsonr(lof_val,  gmm_val)
    scc_val,  _ = spearmanr(lof_val, gmm_val)
    pcc_test, _ = pearsonr(lof_test,  gmm_test)
    scc_test, _ = spearmanr(lof_test, gmm_test)

    # ── [0,0]: Score 산점도 ────────────────────────────────────
    ax = axes[0, 0]
    normal_mask  = val_labels == 0
    anomaly_mask = val_labels == 1
    rng  = np.random.default_rng(0)
    samp = rng.choice(normal_mask.sum(), size=min(5000, normal_mask.sum()), replace=False)

    ax.scatter(lof_val[normal_mask][samp], gmm_val[normal_mask][samp],
               c="steelblue", s=4, alpha=0.3, label="Val 정상")
    ax.scatter(lof_val[anomaly_mask], gmm_val[anomaly_mask],
               c="tomato", s=16, alpha=0.85, label="Val 이상")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="y=x (완전 일치)")
    ax.set_xlabel("LOF Rank Score")
    ax.set_ylabel("GMM Rank Score")
    ax.set_title(
        f"Score 산점도 (val)\n"
        f"Pearson r={pcc_val:.3f}  |  Spearman ρ={scc_val:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=8, markerscale=3)

    # ── [0,1]: 상관계수 히트맵 ────────────────────────────────
    ax = axes[0, 1]
    corr_data = np.array([
        [1.0,     pcc_val,  pearsonr(lof_val, ens_val)[0]],
        [pcc_val, 1.0,      pearsonr(gmm_val, ens_val)[0]],
        [pearsonr(lof_val, ens_val)[0], pearsonr(gmm_val, ens_val)[0], 1.0],
    ])
    im = ax.imshow(corr_data, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    labels_h = ["LOF", "GMM", "Ensemble"]
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels_h)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(labels_h)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr_data[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, color="black")
    ax.set_title(
        f"Pearson 상관계수 히트맵 (val)\n"
        f"val LOF↔GMM  r={pcc_val:.3f}, ρ={scc_val:.3f}  |  "
        f"test  r={pcc_test:.3f}, ρ={scc_test:.3f}",
        fontsize=9,
    )

    # ── [1,0]: 유형별 AUPR 막대 비교 ─────────────────────────
    ax = axes[1, 0]
    types = ["Overall", "Point\n(1~5)", "Contextual\n(6~200)", "Collective\n(201+)"]

    def _type_aupr(scores, labels):
        return [
            evaluate_aupr(scores, labels),
            anomaly_type_aupr(scores, labels, *POINT_LEN),
            anomaly_type_aupr(scores, labels, *CONTEXTUAL_LEN),
            anomaly_type_aupr(scores, labels, *COLLECTIVE_LEN),
        ]

    lof_auprs = _type_aupr(lof_val, val_labels)
    gmm_auprs = _type_aupr(gmm_val, val_labels)
    ens_auprs = _type_aupr(ens_val, val_labels)

    x = np.arange(len(types))
    bw = 0.25
    b_lof = ax.bar(x - bw, lof_auprs, bw, label="LOF-013", color="steelblue", alpha=0.8)
    b_gmm = ax.bar(x,      gmm_auprs, bw, label="GMM-005", color="orange",    alpha=0.8)
    b_ens = ax.bar(x + bw, ens_auprs, bw, label="Ensemble", color="seagreen",  alpha=0.8)

    for bars in [b_lof, b_gmm, b_ens]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=9)
    ax.set_ylabel("AUPR")
    ax.set_ylim(0, min(1.25, max(filter(lambda v: not np.isnan(v),
                                        lof_auprs + gmm_auprs + ens_auprs)) * 1.35))
    ax.set_title("Val 유형별 AUPR 비교 (LOF-013 / GMM-005 / Ensemble)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # ── [1,1]: Score KDE (정상 vs 이상) ─────────────────────
    ax = axes[1, 1]
    _kde_lof = viz_lof_val if viz_lof_val is not None else lof_val
    _kde_gmm = viz_gmm_val if viz_gmm_val is not None else gmm_val
    _kde_ens = viz_ens_val if viz_ens_val is not None else ens_val
    for scores, name, color in [
        (_kde_lof, "LOF-013", "steelblue"),
        (_kde_gmm, "GMM-005", "orange"),
        (_kde_ens, "Ensemble", "seagreen"),
    ]:
        xs = np.linspace(0, 1, 300)
        try:
            ax.plot(xs, gaussian_kde(scores[val_labels == 0])(xs),
                    lw=1.5, ls="--", color=color, alpha=0.6, label=f"{name} 정상")
            ax.plot(xs, gaussian_kde(scores[val_labels == 1])(xs),
                    lw=2.0, ls="-",  color=color, alpha=0.9, label=f"{name} 이상")
        except Exception:
            pass
    score_label = "Minmax Score" if VIZ_MINMAX else "Rank Score"
    ax.set_xlabel(score_label)
    ax.set_ylabel("KDE 밀도")
    ax.set_title(f"Score KDE ({score_label}): 정상(점선) vs 이상(실선) — val", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")


def plot_score_compare(
    scores_dict: dict[str, np.ndarray],
    val_labels:  np.ndarray,
    save_path:   str,
) -> None:
    n = len(scores_dict)
    fig, axes = plt.subplots(n, 1, figsize=(18, 4 * n))
    if n == 1:
        axes = [axes]
    for ax, (name, scores) in zip(axes, scores_dict.items()):
        aupr  = evaluate_aupr(scores,  val_labels)
        auroc = evaluate_auroc(scores, val_labels)
        _draw_full(ax, scores, val_labels,
                   f"[val] {name}   AUROC={auroc:.4f}  AUPR={aupr:.4f}")
    fig.suptitle("모델별 Val Anomaly Score 추이 비교", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  저장: {save_path}")


# ── 리포트용 publication-quality 시각화 ─────────────────────────

def plot_report(
    ens_val_scores:  np.ndarray,
    val_labels:      np.ndarray,
    ens_test_scores: np.ndarray,
    test_labels:     np.ndarray,
    lof_val_scores:  np.ndarray,
    gmm_val_scores:  np.ndarray,
    lof_test_scores: np.ndarray,
    gmm_test_scores: np.ndarray,
    save_path:       str,
) -> None:
    """논문/보고서용 통합 성능 요약 Figure.

    Layout (6열 GridSpec)
    ─────────────────────
    Row 0 (cols 0-1): ROC Curve  (test only · LOF / GMM / Ensemble)
    Row 0 (cols 2-3): PR Curve   (test only · LOF / GMM / Ensemble)
    Row 0 (cols 4-5): Score Hist + KDE  (test, 로그 y축)
    Row 1 (cols 0-5): Test Score Trace  (전체 시계열, 전폭)
    Row 2 (cols 0-5): Test 유형별 확대 패널 (Point / Contextual / Collective)
    """
    RC = {
        "font.family":      "Malgun Gothic",
        "axes.unicode_minus": False,
        "font.size":        12,
        "axes.titlesize":   13,
        "axes.labelsize":   12,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "lines.linewidth":  2.0,
        "axes.spines.top":  False,
        "axes.spines.right": False,
    }
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(24, 18))
        gs  = fig.add_gridspec(
            3, 6,
            height_ratios=[3, 2.5, 2.5],
            hspace=0.42, wspace=0.38,
        )

        # ── [0,0-1] ROC Curve (test only) ───────────────────────
        ax_roc = fig.add_subplot(gs[0, 0:2])
        _TEST_PALETTE = [
            ("LOF-013",  lof_test_scores,  "#4C72B0", "--", 1.8),
            ("GMM-005",  gmm_test_scores,  "#DD8452", "--", 1.8),
            ("Ensemble", ens_test_scores,  "#55A868",  "-", 2.8),
        ]
        for label, scores, color, ls, lw in _TEST_PALETTE:
            fpr, tpr, _ = roc_curve(test_labels, scores)
            auc = evaluate_auroc(scores, test_labels)
            ax_roc.plot(fpr, tpr, color=color, ls=ls, lw=lw,
                        label=f"{label}  AUC={auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.4, label="Random")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve  [test]")
        ax_roc.legend(fontsize=9.5, loc="lower right")
        ax_roc.set_xlim(-0.01, 1.01); ax_roc.set_ylim(-0.01, 1.01)
        ax_roc.grid(alpha=0.25)

        # ── [0,2-3] PR Curve (test only) ────────────────────────
        ax_pr = fig.add_subplot(gs[0, 2:4])
        for label, scores, color, ls, lw in _TEST_PALETTE:
            prec, rec, _ = precision_recall_curve(test_labels, scores)
            aupr = evaluate_aupr(scores, test_labels)
            ax_pr.plot(rec, prec, color=color, ls=ls, lw=lw,
                       label=f"{label}  AUPR={aupr:.3f}")
        baseline = test_labels.mean()
        ax_pr.axhline(baseline, color="gray", ls="--", lw=1.0,
                      label=f"Random (= {baseline:.3f})")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision–Recall Curve  [test]")
        ax_pr.legend(fontsize=9.5, loc="upper right")
        ax_pr.set_xlim(-0.01, 1.01); ax_pr.set_ylim(-0.01, 1.05)
        ax_pr.grid(alpha=0.25)

        # ── [0,4-5] Score 분포 Histogram + KDE (test) ────────────
        ax_hist = fig.add_subplot(gs[0, 4:6])
        normal  = ens_test_scores[test_labels == 0]
        anomaly = ens_test_scores[test_labels == 1]
        bins = np.linspace(0, 1, 80)

        ax_hist.hist(normal,  bins=bins, alpha=0.45, color="#4C72B0",
                     label=f"정상  (n={len(normal):,})", density=True)
        ax_hist.hist(anomaly, bins=bins, alpha=0.55, color="#C44E52",
                     label=f"이상  (n={len(anomaly):,})", density=True)

        xs = np.linspace(0, 1, 400)
        try:
            ax_hist.plot(xs, gaussian_kde(normal)(xs),  lw=2.2, color="#2C5F8A", ls="-")
            ax_hist.plot(xs, gaussian_kde(anomaly)(xs), lw=2.2, color="#8B1A1A", ls="-")
        except Exception:
            pass

        auroc_t = evaluate_auroc(ens_test_scores, test_labels)
        aupr_t  = evaluate_aupr(ens_test_scores,  test_labels)
        ax_hist.set_title(f"Score 분포  [test]\nAUROC={auroc_t:.4f}   AUPR={aupr_t:.4f}")
        ax_hist.set_xlabel("Anomaly Score (Ensemble)")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=9)
        ax_hist.set_xlim(-0.02, 1.02)
        ax_hist.grid(alpha=0.25)

        # ── [1, 0-5] Test Score Trace (전폭) ─────────────────────
        ax_trace = fig.add_subplot(gs[1, :])
        segs = _find_segments(test_labels)
        first = True
        for s, e in segs:
            ax_trace.axvspan(s, e, alpha=0.22, color="#C44E52",
                             label="이상 구간 (정답)" if first else "_nolegend_")
            first = False
        ax_trace.plot(ens_test_scores, lw=0.9, color="#2C5F8A",
                      alpha=0.85, label="Ensemble Score")
        ax_trace.set_xlim(0, len(ens_test_scores))
        ax_trace.set_ylim(-0.02, 1.08)
        ax_trace.set_xlabel("Timestep")
        ax_trace.set_ylabel("Anomaly Score")
        ax_trace.set_title(
            f"Test Score 전체 추이  —  Ensemble (LOF-013 + GMM-005)"
            f"   AUROC={auroc_t:.4f}   AUPR={aupr_t:.4f}"
        )
        ax_trace.legend(loc="upper right", fontsize=9)
        ax_trace.grid(alpha=0.18)

        # ── [2, *] 이상 유형별 확대 패널 ─────────────────────────
        TYPE_META = [
            ("Point",       1,   5,   "#9467BD", 80),
            ("Contextual",  6,   200, "#8C564B", 120),
            ("Collective",  201, 10**9, "#E377C2", 200),
        ]
        col_ptr = 0
        for type_name, lo, hi, color, ctx in TYPE_META:
            matching = [(s, e) for s, e in segs if lo <= (e - s) <= hi]
            n_match = len(matching)
            if n_match == 0:
                continue
            n_cols_use = min(n_match, 2) if n_match <= 4 else 2
            # 각 유형 당 2열씩 할당 (최대 3유형 × 2열 = 6열)
            col_end = col_ptr + 2
            ax_z = fig.add_subplot(gs[2, col_ptr:col_end])
            seg = matching[0]
            s, e = seg
            lo_t = max(0, s - ctx)
            hi_t = min(len(ens_test_scores), e + ctx)
            first_z = True
            for ss, se in _find_segments(test_labels[lo_t:hi_t]):
                ax_z.axvspan(lo_t + ss, lo_t + se, alpha=0.28, color="#C44E52",
                             label="이상 구간" if first_z else "_nolegend_")
                first_z = False
            ax_z.plot(np.arange(lo_t, hi_t), ens_test_scores[lo_t:hi_t],
                      lw=1.4, color=color)
            seg_len = e - s
            ax_z.set_title(
                f"{type_name}  (len={seg_len}, t={s}~{e-1})\n"
                f"AUPR={evaluate_aupr(ens_test_scores[lo_t:hi_t], test_labels[lo_t:hi_t]):.3f}",
                fontsize=10,
            )
            ax_z.set_xlabel("Timestep", fontsize=9)
            ax_z.set_ylabel("Score", fontsize=9)
            ax_z.set_xlim(lo_t, hi_t)
            ax_z.set_ylim(-0.02, 1.08)
            ax_z.legend(fontsize=8)
            ax_z.grid(alpha=0.2)
            col_ptr = col_end

        fig.suptitle(
            "앙상블 009  —  LOF-013 + GMM-005  성능 요약",
            fontsize=16, fontweight="bold", y=0.995,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  저장: {save_path}")


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    print("=== 앙상블 009: LOF-013 + GMM-005 (Rank 평균) ===\n")

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
    disc_cols = [c for c in x_cols if c in    DISCRETE_COLS]
    print(f"  연속형: {len(cont_cols)}채널  이산형: {len(disc_cols)}채널")

    # ── LOF (LOF-013) ─────────────────────────────────────────────
    print("\n[LOF-013] 파이프라인 구성 중...")
    lof_train_X, lof_val_X, lof_test_X = build_lof_features(
        train_df, val_df, test_df, cont_cols, disc_cols
    )
    print(f"  LOF feature dim: {lof_train_X.shape[1]}"
          f"  (x_f8 diff → cont {len(cont_cols)} scaler + disc {len(disc_cols)} ratio W={LOF_DISC_WINDOW})")

    print("  LOF 학습 중...")
    lof_model = fit_lof(lof_train_X, n_neighbors=LOF_NEIGHBORS, contamination=LOF_CONTAM)
    raw_lof_val     = flip_score(lof_model.score_samples(lof_val_X))
    raw_lof_test    = flip_score(lof_model.score_samples(lof_test_X))
    lof_val_scores  = rank_normalize(raw_lof_val)
    lof_test_scores = rank_normalize(raw_lof_test)
    print(f"  LOF val  AUROC={evaluate_auroc(lof_val_scores,  val_labels):.4f}"
          f"  AUPR={evaluate_aupr(lof_val_scores,  val_labels):.4f}")
    print(f"  LOF test AUROC={evaluate_auroc(lof_test_scores, test_labels):.4f}"
          f"  AUPR={evaluate_aupr(lof_test_scores, test_labels):.4f}")

    # ── GMM (GMM-005) ─────────────────────────────────────────────
    print("\n[GMM-005] 파이프라인 구성 중...")
    gmm_train_X, gmm_val_X, gmm_test_X, pca = build_gmm_features(
        train_df, val_df, test_df, cont_cols
    )
    print(f"  GMM feature dim: {gmm_train_X.shape[1]}"
          f"  (x_f8 diff → Scaler → PCA({GMM_PCA_VAR:.0%}) → Rolling W={GMM_WINDOW} × {len(GMM_STATS)}통계)")

    print(f"  GMM 학습 중... (n_components={GMM_N_COMP}, covariance_type={GMM_COV_TYPE!r})")
    gmm_model = fit_gmm(gmm_train_X, n_components=GMM_N_COMP, covariance_type=GMM_COV_TYPE)
    raw_gmm_val     = flip_score(gmm_model.score_samples(gmm_val_X))
    raw_gmm_test    = flip_score(gmm_model.score_samples(gmm_test_X))
    gmm_val_scores  = rank_normalize(raw_gmm_val)
    gmm_test_scores = rank_normalize(raw_gmm_test)
    print(f"  GMM val  AUROC={evaluate_auroc(gmm_val_scores,  val_labels):.4f}"
          f"  AUPR={evaluate_aupr(gmm_val_scores,  val_labels):.4f}")
    print(f"  GMM test AUROC={evaluate_auroc(gmm_test_scores, test_labels):.4f}"
          f"  AUPR={evaluate_aupr(gmm_test_scores, test_labels):.4f}")

    # ── 앙상블 ────────────────────────────────────────────────────
    print("\n[Ensemble] rank 평균 앙상블...")
    ens_val_scores  = ensemble_scores({"lof": lof_val_scores,  "gmm": gmm_val_scores})
    ens_test_scores = ensemble_scores({"lof": lof_test_scores, "gmm": gmm_test_scores})

    # ── 상관계수 출력 ─────────────────────────────────────────────
    pcc_val,  _ = pearsonr(lof_val_scores,  gmm_val_scores)
    scc_val,  _ = spearmanr(lof_val_scores, gmm_val_scores)
    pcc_test, _ = pearsonr(lof_test_scores,  gmm_test_scores)
    scc_test, _ = spearmanr(lof_test_scores, gmm_test_scores)
    print(f"\n[다양성] LOF-013 ↔ GMM-005 score 상관계수")
    print(f"  val  : Pearson r={pcc_val:.4f}  Spearman ρ={scc_val:.4f}")
    print(f"  test : Pearson r={pcc_test:.4f}  Spearman ρ={scc_test:.4f}")
    if abs(scc_val) < 0.5:
        print("  → 상관이 낮아 앙상블 효과 기대 ✓")
    elif abs(scc_val) < 0.75:
        print("  → 중간 상관 — 앙상블 이득 보통")
    else:
        print("  → 상관이 높아 앙상블 이득 제한적")

    # ── 최종 평가 표 ──────────────────────────────────────────────
    print(f"\n[최종 평가]")
    header = f"  {'모델':<12} {'val AUROC':>10} {'val AUPR':>9} {'test AUROC':>11} {'test AUPR':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, vs, vl, ts, tl in [
        ("LOF-013",  lof_val_scores, val_labels,  lof_test_scores, test_labels),
        ("GMM-005",  gmm_val_scores, val_labels,  gmm_test_scores, test_labels),
        ("Ensemble", ens_val_scores, val_labels,  ens_test_scores, test_labels),
    ]:
        print(f"  {name:<12} {evaluate_auroc(vs, vl):>10.4f} {evaluate_aupr(vs, vl):>9.4f}"
              f" {evaluate_auroc(ts, tl):>11.4f} {evaluate_aupr(ts, tl):>9.4f}")

    print(f"\n  [Val] 유형별 AUPR")
    print(f"  {'유형':<14} {'LOF-013':>9} {'GMM-005':>9} {'Ensemble':>10}")
    print("  " + "-" * 45)
    for type_name, lo, hi in [
        ("Point(1~5)",  *POINT_LEN),
        ("Contextual",  *CONTEXTUAL_LEN),
        ("Collective",  *COLLECTIVE_LEN),
    ]:
        la   = anomaly_type_aupr(lof_val_scores, val_labels, lo, hi)
        ga   = anomaly_type_aupr(gmm_val_scores, val_labels, lo, hi)
        ensa = anomaly_type_aupr(ens_val_scores, val_labels, lo, hi)
        def _fmt(v): return f"{v:.4f}" if not np.isnan(v) else "   nan"
        print(f"  {type_name:<14} {_fmt(la):>9} {_fmt(ga):>9} {_fmt(ensa):>10}")

    print(f"\n  [Test] 유형별 AUPR")
    print(f"  {'유형':<14} {'LOF-013':>9} {'GMM-005':>9} {'Ensemble':>10}")
    print("  " + "-" * 45)
    for type_name, lo, hi in [
        ("Point(1~5)",  *POINT_LEN),
        ("Contextual",  *CONTEXTUAL_LEN),
        ("Collective",  *COLLECTIVE_LEN),
    ]:
        la   = anomaly_type_aupr(lof_test_scores, test_labels, lo, hi)
        ga   = anomaly_type_aupr(gmm_test_scores, test_labels, lo, hi)
        ensa = anomaly_type_aupr(ens_test_scores, test_labels, lo, hi)
        def _fmt(v): return f"{v:.4f}" if not np.isnan(v) else "   nan"
        print(f"  {type_name:<14} {_fmt(la):>9} {_fmt(ga):>9} {_fmt(ensa):>10}")

    # ── 시각화용 점수 변환 ────────────────────────────────────────
    if VIZ_MINMAX:
        viz_lof_val  = minmax(raw_lof_val)
        viz_gmm_val  = minmax(raw_gmm_val)
        viz_ens_val  = minmax(minmax(raw_lof_val)  + minmax(raw_gmm_val))
        viz_ens_test = minmax(minmax(raw_lof_test) + minmax(raw_gmm_test))
        print("\n  [시각화] minmax 적용 — 개별: raw flip score, 앙상블: minmax 합산 후 재정규화 (평가 지표 불변)")
    else:
        viz_lof_val  = lof_val_scores
        viz_gmm_val  = gmm_val_scores
        viz_ens_val  = ens_val_scores
        viz_ens_test = ens_test_scores

    # ── 시각화 ────────────────────────────────────────────────────
    plot_diversity(
        lof_val_scores, gmm_val_scores, val_labels,
        lof_test_scores, gmm_test_scores, test_labels,
        ens_val_scores,
        save_path=str(OUTPUT_DIR / "009_diversity.png"),
        viz_lof_val=viz_lof_val,
        viz_gmm_val=viz_gmm_val,
        viz_ens_val=viz_ens_val,
    )

    plot_score_compare(
        {
            f"LOF-013 (x_f8 diff → cont {len(cont_cols)}ch scaler + disc {len(disc_cols)}ch ratio W={LOF_DISC_WINDOW})": viz_lof_val,
            f"GMM-005 (x_f8 diff + PCA + rolling W={GMM_WINDOW})":                                                       viz_gmm_val,
            "Ensemble (rank 평균)":                                                                                        viz_ens_val,
        },
        val_labels,
        save_path=str(OUTPUT_DIR / "009_score_trace.png"),
    )

    plot_full(viz_ens_val, val_labels, viz_ens_test, test_labels,
              tag="009_ensemble_lof013_gmm005",
              save_path=str(OUTPUT_DIR / "009_val_score_trace.png"))
    plot_zooms(viz_ens_val, val_labels, viz_ens_test, test_labels,
               tag="009_ensemble_lof013_gmm005",
               save_path=str(OUTPUT_DIR / "009_val_score_zoom.png"))
    plot_score_hist(viz_ens_val, val_labels, viz_ens_test, test_labels,
                    tag="009_ensemble_lof013_gmm005",
                    save_path=str(OUTPUT_DIR / "009_score_hist.png"))

    plot_report(
        ens_val_scores,  val_labels,
        ens_test_scores, test_labels,
        lof_val_scores,  gmm_val_scores,
        lof_test_scores, gmm_test_scores,
        save_path=str(OUTPUT_DIR / "009_report.png"),
    )

    print("\n=== 완료 ===")