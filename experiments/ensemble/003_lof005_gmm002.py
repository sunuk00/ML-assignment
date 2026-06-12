"""앙상블 003: LOF-005 (Raw Cont + Scaler) + GMM-002 (PCA Rolling) — Rank 평균 앙상블

모델 조합
---------
  LOF: 연속형 7채널 원본 → StandardScaler
       → LOF-005 기반 파이프라인 (시간 맥락 없음, 현재 시점 국소 밀도)
  GMM: 연속형 7채널 → StandardScaler → PCA(95%) → Rolling(W=30) → 5통계
       → GMM-002 기반 파이프라인 (PCA 공간에서 시간 맥락 포함 전역 밀도)

설계 근거
---------
  LOF: 현재 시점의 원본 피처 공간에서 국소(local) 밀도 이상 탐지.
  GMM: PCA로 차원 축소 후 rolling 통계를 통해 시간 맥락까지 반영한 전역(global) 밀도 추정.
  두 모델은 (1) 시간 정보 활용 여부, (2) 밀도 추정 방식(국소 vs 전역)이 모두 달라
  다양성이 높아 앙상블 이득이 기대됩니다.

앙상블 전략
-----------
  rank_normalize 후 단순 평균 (equal weight)

출력 파일
---------
  003_diversity.png       — Score 산점도 / 상관계수 히트맵 / AUPR 비교 / Score KDE
  003_score_trace.png     — val: LOF / GMM / Ensemble 3행 score 추이
  003_val_score_trace.png — Ensemble만: val+test 전체 추이
  003_val_score_zoom.png  — Ensemble: 이상 구간 확대
  003_score_hist.png      — Ensemble: score 분포 히스토그램
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

from src.data_loader   import load_data
from src.preprocessing import fill_missing, fit_scaler, apply_scaler, fit_pca, apply_pca
from src.features      import DISCRETE_COLS, rolling_features
from src.models        import fit_lof, fit_gmm
from src.ensemble      import flip_score, rank_normalize, ensemble_scores
from src.evaluate      import (evaluate_aupr, evaluate_auroc, anomaly_type_aupr,
                               plot_full, plot_zooms, plot_score_hist, _draw_full)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "ensemble" / "outputs"

# ── LOF 하이퍼파라미터 (LOF-005 기반) ───────────────────────────
LOF_NEIGHBORS = 10
LOF_CONTAM    = 0.0001

# ── GMM 하이퍼파라미터 (GMM-002 기반) ───────────────────────────
GMM_WINDOW      = 30
GMM_ROLL_STATS  = ["mean", "std", "min", "max", "range"]
GMM_N_COMP      = 5
GMM_COV_TYPE    = "full"
GMM_PCA_VAR     = 0.95

POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)

# 앙상블 조합은 rank_normalize 고정.
# VIZ_MINMAX=True: 시각화 직전에만 minmax 재정규화 (AUPR/AUROC 불변)
VIZ_MINMAX = True


def minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


# ── LOF 피처 빌더 (LOF-005: raw cont + scaler) ──────────────────

def build_lof005_features(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    cont_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler    = fit_scaler(train_df[cont_cols])
    tr_scaled = apply_scaler(scaler, train_df[cont_cols])
    va_scaled = apply_scaler(scaler, val_df[cont_cols])
    te_scaled = apply_scaler(scaler, test_df[cont_cols])
    return tr_scaled.to_numpy(), va_scaled.to_numpy(), te_scaled.to_numpy()


# ── GMM 피처 빌더 (GMM-002: scaler → PCA → rolling) ────────────

def build_gmm002_features(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    cont_cols: list[str],
    w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    scaler       = fit_scaler(train_df[cont_cols])
    train_scaled = apply_scaler(scaler, train_df[cont_cols])
    val_scaled   = apply_scaler(scaler, val_df[cont_cols])
    test_scaled  = apply_scaler(scaler, test_df[cont_cols])

    pca       = fit_pca(train_scaled, variance=GMM_PCA_VAR)
    train_pca = apply_pca(pca, train_scaled)
    val_pca   = apply_pca(pca, val_scaled)
    test_pca  = apply_pca(pca, test_scaled)

    pc_cols   = list(train_pca.columns)
    train_X   = rolling_features(train_pca, cols=pc_cols, window_size=w, stats=GMM_ROLL_STATS).to_numpy()
    val_X     = rolling_features(val_pca,   cols=pc_cols, window_size=w, stats=GMM_ROLL_STATS).to_numpy()
    test_X    = rolling_features(test_pca,  cols=pc_cols, window_size=w, stats=GMM_ROLL_STATS).to_numpy()

    return train_X, val_X, test_X, pca


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
    fig.suptitle("앙상블 003 — 모델 다양성 진단 (LOF-005 × GMM-002)", fontsize=13)

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

    x  = np.arange(len(types))
    bw = 0.25
    b_lof = ax.bar(x - bw, lof_auprs, bw, label="LOF-005", color="orange",    alpha=0.8)
    b_gmm = ax.bar(x,      gmm_auprs, bw, label="GMM-002", color="mediumpurple", alpha=0.8)
    b_ens = ax.bar(x + bw, ens_auprs, bw, label="Ensemble", color="seagreen", alpha=0.8)

    for bars in [b_lof, b_gmm, b_ens]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=9)
    ax.set_ylabel("AUPR")
    all_vals = [v for v in lof_auprs + gmm_auprs + ens_auprs if not np.isnan(v)]
    ax.set_ylim(0, min(1.25, max(all_vals) * 1.35) if all_vals else 1.0)
    ax.set_title("Val 유형별 AUPR 비교 (LOF-005 / GMM-002 / Ensemble)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # ── [1,1]: Score KDE (정상 vs 이상) ─────────────────────
    ax = axes[1, 1]
    _kde_lof = viz_lof_val if viz_lof_val is not None else lof_val
    _kde_gmm = viz_gmm_val if viz_gmm_val is not None else gmm_val
    _kde_ens = viz_ens_val if viz_ens_val is not None else ens_val
    for scores, name, color in [
        (_kde_lof, "LOF-005",  "orange"),
        (_kde_gmm, "GMM-002",  "mediumpurple"),
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


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    print("=== 앙상블 003: LOF-005 + GMM-002 (Rank 평균) ===\n")

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
    print(f"  연속형: {len(cont_cols)}채널 (이산형 제외)")

    # ── LOF (LOF-005) ─────────────────────────────────────────────
    print("\n[LOF-005] 파이프라인 구성 중...")
    lof_train_X, lof_val_X, lof_test_X = build_lof005_features(
        train_df, val_df, test_df, cont_cols
    )
    print(f"  LOF feature dim: {lof_train_X.shape[1]}  "
          f"(cont {len(cont_cols)} raw + StandardScaler)")

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

    # ── GMM (GMM-002) ─────────────────────────────────────────────
    print("\n[GMM-002] 파이프라인 구성 중...")
    gmm_train_X, gmm_val_X, gmm_test_X, pca = build_gmm002_features(
        train_df, val_df, test_df, cont_cols, GMM_WINDOW
    )
    print(f"  GMM feature dim: {gmm_train_X.shape[1]}  "
          f"(PCA {pca.n_components_}성분 × {len(GMM_ROLL_STATS)} stats, W={GMM_WINDOW})")

    print(f"  GMM 학습 중... (n_components={GMM_N_COMP}, cov={GMM_COV_TYPE!r})")
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
    pcc_test, _ = pearsonr(lof_test_scores, gmm_test_scores)
    scc_test, _ = spearmanr(lof_test_scores, gmm_test_scores)
    print(f"\n[다양성] LOF-005 ↔ GMM-002 score 상관계수")
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
        ("LOF-005",  lof_val_scores, val_labels, lof_test_scores, test_labels),
        ("GMM-002",  gmm_val_scores, val_labels, gmm_test_scores, test_labels),
        ("Ensemble", ens_val_scores, val_labels, ens_test_scores, test_labels),
    ]:
        print(f"  {name:<12} {evaluate_auroc(vs, vl):>10.4f} {evaluate_aupr(vs, vl):>9.4f}"
              f" {evaluate_auroc(ts, tl):>11.4f} {evaluate_aupr(ts, tl):>9.4f}")

    print(f"\n  [Val] 유형별 AUPR")
    print(f"  {'유형':<14} {'LOF-005':>9} {'GMM-002':>9} {'Ensemble':>10}")
    print("  " + "-" * 45)
    for type_name, lo, hi in [
        ("Point(1~5)",  *POINT_LEN),
        ("Contextual",  *CONTEXTUAL_LEN),
        ("Collective",  *COLLECTIVE_LEN),
    ]:
        lfa  = anomaly_type_aupr(lof_val_scores, val_labels, lo, hi)
        gfa  = anomaly_type_aupr(gmm_val_scores, val_labels, lo, hi)
        ensa = anomaly_type_aupr(ens_val_scores, val_labels, lo, hi)
        def _fmt(v): return f"{v:.4f}" if not np.isnan(v) else "   nan"
        print(f"  {type_name:<14} {_fmt(lfa):>9} {_fmt(gfa):>9} {_fmt(ensa):>10}")

    print(f"\n  [Test] 유형별 AUPR")
    print(f"  {'유형':<14} {'LOF-005':>9} {'GMM-002':>9} {'Ensemble':>10}")
    print("  " + "-" * 45)
    for type_name, lo, hi in [
        ("Point(1~5)",  *POINT_LEN),
        ("Contextual",  *CONTEXTUAL_LEN),
        ("Collective",  *COLLECTIVE_LEN),
    ]:
        lfa  = anomaly_type_aupr(lof_test_scores, test_labels, lo, hi)
        gfa  = anomaly_type_aupr(gmm_test_scores, test_labels, lo, hi)
        ensa = anomaly_type_aupr(ens_test_scores, test_labels, lo, hi)
        def _fmt(v): return f"{v:.4f}" if not np.isnan(v) else "   nan"
        print(f"  {type_name:<14} {_fmt(lfa):>9} {_fmt(gfa):>9} {_fmt(ensa):>10}")

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
    # plot_diversity 산점도·히트맵은 rank 점수 그대로 (상관 구조 보존)
    plot_diversity(
        lof_val_scores, gmm_val_scores, val_labels,
        lof_test_scores, gmm_test_scores, test_labels,
        ens_val_scores,
        save_path=str(OUTPUT_DIR / "003_diversity.png"),
        viz_lof_val=viz_lof_val,
        viz_gmm_val=viz_gmm_val,
        viz_ens_val=viz_ens_val,
    )

    plot_score_compare(
        {
            "LOF-005 (raw cont + scaler)": viz_lof_val,
            "GMM-002 (PCA rolling W=30)":  viz_gmm_val,
            "Ensemble (rank 평균)":         viz_ens_val,
        },
        val_labels,
        save_path=str(OUTPUT_DIR / "003_score_trace.png"),
    )

    plot_full(viz_ens_val, val_labels, viz_ens_test, test_labels,
              tag="003_ensemble_lof005_gmm002",
              save_path=str(OUTPUT_DIR / "003_val_score_trace.png"))
    plot_zooms(viz_ens_val, val_labels, viz_ens_test, test_labels,
               tag="003_ensemble_lof005_gmm002",
               save_path=str(OUTPUT_DIR / "003_val_score_zoom.png"))
    plot_score_hist(viz_ens_val, val_labels, viz_ens_test, test_labels,
                    tag="003_ensemble_lof005_gmm002",
                    save_path=str(OUTPUT_DIR / "003_score_hist.png"))

    print("\n=== 완료 ===")
