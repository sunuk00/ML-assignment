"""앙상블 005: LOF-005 + GMM-002 + IF-013 — Rank 평균 앙상블

LOF: 연속형 7채널 원본 → StandardScaler (LOF-005 기반)
GMM: 연속형 7채널 → StandardScaler → PCA → Rolling 통계 (GMM-002 기반)
IF: x_f8 차분 → 연속형 Rolling 통계 + 이산형 Rolling Mean (IF-013 기반)
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

from src.data_loader   import load_data
from src.preprocessing import fill_missing, fit_scaler, apply_scaler, fit_pca, apply_pca
from src.features      import DISCRETE_COLS, rolling_features
from src.models        import fit_lof, fit_gmm, fit_isolation_forest
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

# ── IF 하이퍼파라미터 (IF-013 기반) ─────────────────────────────
IF_DIFF_COL   = "x_f8"
IF_WINDOW     = 100
IF_STATS      = ["mean", "std", "min", "max", "range"]
IF_ESTIMATORS = 300
IF_CONTAM     = 0.0001
IF_RANDOM     = 42

POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)

VIZ_MINMAX = True


def minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


# ── LOF 피처 빌더 (LOF-005: raw cont + scaler) ──────────────────

def build_lof005_features(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    test_df:   pd.DataFrame,
    cont_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler    = fit_scaler(train_df[cont_cols])
    tr_scaled = apply_scaler(scaler, train_df[cont_cols])
    va_scaled = apply_scaler(scaler, val_df[cont_cols])
    te_scaled = apply_scaler(scaler, test_df[cont_cols])
    return tr_scaled.to_numpy(), va_scaled.to_numpy(), te_scaled.to_numpy()


# ── GMM 피처 빌더 (GMM-002: scaler → PCA → rolling) ────────────

def build_gmm002_features(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    test_df:   pd.DataFrame,
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

    pc_cols  = list(train_pca.columns)
    train_X  = rolling_features(train_pca, cols=pc_cols, window_size=w, stats=GMM_ROLL_STATS).to_numpy()
    val_X    = rolling_features(val_pca,   cols=pc_cols, window_size=w, stats=GMM_ROLL_STATS).to_numpy()
    test_X   = rolling_features(test_pca,  cols=pc_cols, window_size=w, stats=GMM_ROLL_STATS).to_numpy()
    return train_X, val_X, test_X, pca


# ── IF 피처 빌더 (IF-013: x_f8 diff + cont rolling + disc ratio) ─

def _apply_diff(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].diff().fillna(0)
    return df


def build_if013_features(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    test_df:   pd.DataFrame,
    cont_cols: list[str],
    disc_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr_d = _apply_diff(train_df, IF_DIFF_COL)
    va_d = _apply_diff(val_df,   IF_DIFF_COL)
    te_d = _apply_diff(test_df,  IF_DIFF_COL)

    tr_cont = rolling_features(tr_d, cols=cont_cols, window_size=IF_WINDOW, stats=IF_STATS)
    va_cont = rolling_features(va_d, cols=cont_cols, window_size=IF_WINDOW, stats=IF_STATS)
    te_cont = rolling_features(te_d, cols=cont_cols, window_size=IF_WINDOW, stats=IF_STATS)

    tr_disc = tr_d[disc_cols].rolling(IF_WINDOW, min_periods=1).mean().fillna(0)
    va_disc = va_d[disc_cols].rolling(IF_WINDOW, min_periods=1).mean().fillna(0)
    te_disc = te_d[disc_cols].rolling(IF_WINDOW, min_periods=1).mean().fillna(0)

    return (
        np.hstack([tr_cont.to_numpy(), tr_disc.to_numpy()]),
        np.hstack([va_cont.to_numpy(), va_disc.to_numpy()]),
        np.hstack([te_cont.to_numpy(), te_disc.to_numpy()]),
    )


# ── 다양성 분석 시각화 (3모델 × 4x4 히트맵) ─────────────────────

def plot_diversity(
    lof_val:     np.ndarray,
    gmm_val:     np.ndarray,
    if_val:      np.ndarray,
    val_labels:  np.ndarray,
    ens_val:     np.ndarray,
    save_path:   str,
    viz_lof_val: np.ndarray | None = None,
    viz_gmm_val: np.ndarray | None = None,
    viz_if_val:  np.ndarray | None = None,
    viz_ens_val: np.ndarray | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("앙상블 005 — 모델 다양성 진단 (LOF-005 × GMM-002 × IF-013)", fontsize=13)

    # ── [0,0]: 4×4 상관계수 히트맵 ───────────────────────────────
    ax = axes[0, 0]
    model_scores = [lof_val, gmm_val, if_val, ens_val]
    model_names  = ["LOF-005", "GMM-002", "IF-013", "Ensemble"]
    n = len(model_scores)
    corr_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_mat[i, j] = pearsonr(model_scores[i], model_scores[j])[0]

    im = ax.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(n)); ax.set_xticklabels(model_names, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(model_names, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr_mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, color="black")
    pcc_lof_gmm = corr_mat[0, 1]
    pcc_lof_if  = corr_mat[0, 2]
    pcc_gmm_if  = corr_mat[1, 2]
    ax.set_title(
        f"Pearson 상관계수 히트맵 (val)\n"
        f"LOF↔GMM={pcc_lof_gmm:.3f}  LOF↔IF={pcc_lof_if:.3f}  GMM↔IF={pcc_gmm_if:.3f}",
        fontsize=9,
    )

    # ── [0,1]: Score 산점도 (LOF vs GMM) ─────────────────────────
    ax = axes[0, 1]
    normal_mask  = val_labels == 0
    anomaly_mask = val_labels == 1
    rng  = np.random.default_rng(0)
    samp = rng.choice(normal_mask.sum(), size=min(5000, normal_mask.sum()), replace=False)

    ax.scatter(lof_val[normal_mask][samp], gmm_val[normal_mask][samp],
               c="steelblue", s=4, alpha=0.3, label="Val 정상")
    ax.scatter(lof_val[anomaly_mask], gmm_val[anomaly_mask],
               c="tomato", s=16, alpha=0.85, label="Val 이상")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("LOF Rank Score"); ax.set_ylabel("GMM Rank Score")
    scc_lof_gmm = spearmanr(lof_val, gmm_val)[0]
    ax.set_title(f"Score 산점도: LOF vs GMM (val)\nSpearman ρ={scc_lof_gmm:.3f}", fontsize=10)
    ax.legend(fontsize=8, markerscale=3)

    # ── [1,0]: 유형별 AUPR 막대 비교 (4모델) ─────────────────────
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
    if_auprs  = _type_aupr(if_val,  val_labels)
    ens_auprs = _type_aupr(ens_val, val_labels)

    x  = np.arange(len(types))
    bw = 0.18
    b_lof = ax.bar(x - 1.5*bw, lof_auprs, bw, label="LOF-005", color="orange",       alpha=0.8)
    b_gmm = ax.bar(x - 0.5*bw, gmm_auprs, bw, label="GMM-002", color="mediumpurple",  alpha=0.8)
    b_if  = ax.bar(x + 0.5*bw, if_auprs,  bw, label="IF-013",  color="steelblue",     alpha=0.8)
    b_ens = ax.bar(x + 1.5*bw, ens_auprs, bw, label="Ensemble", color="seagreen",     alpha=0.8)

    for bars in [b_lof, b_gmm, b_if, b_ens]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=9)
    ax.set_ylabel("AUPR")
    all_vals = [v for v in lof_auprs + gmm_auprs + if_auprs + ens_auprs if not np.isnan(v)]
    ax.set_ylim(0, min(1.25, max(all_vals) * 1.35) if all_vals else 1.0)
    ax.set_title("Val 유형별 AUPR 비교 (LOF-005 / GMM-002 / IF-013 / Ensemble)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # ── [1,1]: Score KDE ─────────────────────────────────────────
    ax = axes[1, 1]
    _kde_lof = viz_lof_val if viz_lof_val is not None else lof_val
    _kde_gmm = viz_gmm_val if viz_gmm_val is not None else gmm_val
    _kde_if  = viz_if_val  if viz_if_val  is not None else if_val
    _kde_ens = viz_ens_val if viz_ens_val is not None else ens_val
    for scores, name, color in [
        (_kde_lof, "LOF-005",  "orange"),
        (_kde_gmm, "GMM-002",  "mediumpurple"),
        (_kde_if,  "IF-013",   "steelblue"),
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
    ax.set_xlabel(score_label); ax.set_ylabel("KDE 밀도")
    ax.set_title(f"Score KDE ({score_label}): 정상(점선) vs 이상(실선) — val", fontsize=10)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

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
    print("=== 앙상블 005: LOF-005 + GMM-002 + IF-013 (Rank 평균) ===\n")

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

    # ── IF (IF-013) ───────────────────────────────────────────────
    print("\n[IF-013] 파이프라인 구성 중...")
    if_train_X, if_val_X, if_test_X = build_if013_features(
        train_df, val_df, test_df, cont_cols, disc_cols
    )
    print(f"  IF feature dim: {if_train_X.shape[1]}  "
          f"(x_f8 diff → cont {len(cont_cols)}×{len(IF_STATS)} + disc {len(disc_cols)}×ratio, W={IF_WINDOW})")

    print(f"  IF 학습 중... (n_estimators={IF_ESTIMATORS})")
    if_model = fit_isolation_forest(
        if_train_X, n_estimators=IF_ESTIMATORS,
        contamination=IF_CONTAM, random_state=IF_RANDOM,
    )
    raw_if_val     = flip_score(if_model.score_samples(if_val_X))
    raw_if_test    = flip_score(if_model.score_samples(if_test_X))
    if_val_scores  = rank_normalize(raw_if_val)
    if_test_scores = rank_normalize(raw_if_test)
    print(f"  IF  val  AUROC={evaluate_auroc(if_val_scores,  val_labels):.4f}"
          f"  AUPR={evaluate_aupr(if_val_scores,  val_labels):.4f}")
    print(f"  IF  test AUROC={evaluate_auroc(if_test_scores, test_labels):.4f}"
          f"  AUPR={evaluate_aupr(if_test_scores, test_labels):.4f}")

    # ── 앙상블 ────────────────────────────────────────────────────
    print("\n[Ensemble] rank 평균 앙상블...")
    ens_val_scores  = ensemble_scores({"lof": lof_val_scores,  "gmm": gmm_val_scores,  "if": if_val_scores})
    ens_test_scores = ensemble_scores({"lof": lof_test_scores, "gmm": gmm_test_scores, "if": if_test_scores})

    # ── 상관계수 출력 ─────────────────────────────────────────────
    print(f"\n[다양성] 모델 간 score 상관계수 (val, Spearman)")
    pairs = [
        ("LOF-005", lof_val_scores, "GMM-002", gmm_val_scores),
        ("LOF-005", lof_val_scores, "IF-013",  if_val_scores),
        ("GMM-002", gmm_val_scores, "IF-013",  if_val_scores),
    ]
    for n1, s1, n2, s2 in pairs:
        pcc = pearsonr(s1, s2)[0]
        scc = spearmanr(s1, s2)[0]
        print(f"  {n1} ↔ {n2} : Pearson r={pcc:.4f}  Spearman ρ={scc:.4f}")

    # ── 최종 평가 표 ──────────────────────────────────────────────
    print(f"\n[최종 평가]")
    header = f"  {'모델':<12} {'val AUROC':>10} {'val AUPR':>9} {'test AUROC':>11} {'test AUPR':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, vs, vl, ts, tl in [
        ("LOF-005",  lof_val_scores, val_labels,  lof_test_scores, test_labels),
        ("GMM-002",  gmm_val_scores, val_labels,  gmm_test_scores, test_labels),
        ("IF-013",   if_val_scores,  val_labels,  if_test_scores,  test_labels),
        ("Ensemble", ens_val_scores, val_labels,  ens_test_scores, test_labels),
    ]:
        print(f"  {name:<12} {evaluate_auroc(vs, vl):>10.4f} {evaluate_aupr(vs, vl):>9.4f}"
              f" {evaluate_auroc(ts, tl):>11.4f} {evaluate_aupr(ts, tl):>9.4f}")

    print(f"\n  [Val] 유형별 AUPR")
    print(f"  {'유형':<14} {'LOF-005':>9} {'GMM-002':>9} {'IF-013':>8} {'Ensemble':>10}")
    print("  " + "-" * 54)
    for type_name, lo, hi in [
        ("Point(1~5)",  *POINT_LEN),
        ("Contextual",  *CONTEXTUAL_LEN),
        ("Collective",  *COLLECTIVE_LEN),
    ]:
        la   = anomaly_type_aupr(lof_val_scores, val_labels, lo, hi)
        ga   = anomaly_type_aupr(gmm_val_scores, val_labels, lo, hi)
        ia   = anomaly_type_aupr(if_val_scores,  val_labels, lo, hi)
        ensa = anomaly_type_aupr(ens_val_scores, val_labels, lo, hi)
        def _fmt(v): return f"{v:.4f}" if not np.isnan(v) else "   nan"
        print(f"  {type_name:<14} {_fmt(la):>9} {_fmt(ga):>9} {_fmt(ia):>8} {_fmt(ensa):>10}")

    print(f"\n  [Test] 유형별 AUPR")
    print(f"  {'유형':<14} {'LOF-005':>9} {'GMM-002':>9} {'IF-013':>8} {'Ensemble':>10}")
    print("  " + "-" * 54)
    for type_name, lo, hi in [
        ("Point(1~5)",  *POINT_LEN),
        ("Contextual",  *CONTEXTUAL_LEN),
        ("Collective",  *COLLECTIVE_LEN),
    ]:
        la   = anomaly_type_aupr(lof_test_scores, test_labels, lo, hi)
        ga   = anomaly_type_aupr(gmm_test_scores, test_labels, lo, hi)
        ia   = anomaly_type_aupr(if_test_scores,  test_labels, lo, hi)
        ensa = anomaly_type_aupr(ens_test_scores, test_labels, lo, hi)
        def _fmt(v): return f"{v:.4f}" if not np.isnan(v) else "   nan"
        print(f"  {type_name:<14} {_fmt(la):>9} {_fmt(ga):>9} {_fmt(ia):>8} {_fmt(ensa):>10}")

    # ── 시각화용 점수 변환 ────────────────────────────────────────
    if VIZ_MINMAX:
        viz_lof_val  = minmax(raw_lof_val)
        viz_gmm_val  = minmax(raw_gmm_val)
        viz_if_val   = minmax(raw_if_val)
        viz_ens_val  = minmax(minmax(raw_lof_val)  + minmax(raw_gmm_val)  + minmax(raw_if_val))
        viz_ens_test = minmax(minmax(raw_lof_test) + minmax(raw_gmm_test) + minmax(raw_if_test))
        print("\n  [시각화] minmax 적용 — 개별: raw flip score, 앙상블: minmax 합산 후 재정규화 (평가 지표 불변)")
    else:
        viz_lof_val  = lof_val_scores
        viz_gmm_val  = gmm_val_scores
        viz_if_val   = if_val_scores
        viz_ens_val  = ens_val_scores
        viz_ens_test = ens_test_scores

    # ── 시각화 ────────────────────────────────────────────────────
    plot_diversity(
        lof_val_scores, gmm_val_scores, if_val_scores, val_labels,
        ens_val_scores,
        save_path=str(OUTPUT_DIR / "005_diversity.png"),
        viz_lof_val=viz_lof_val,
        viz_gmm_val=viz_gmm_val,
        viz_if_val=viz_if_val,
        viz_ens_val=viz_ens_val,
    )

    plot_score_compare(
        {
            "LOF-005 (raw cont + scaler)":               viz_lof_val,
            f"GMM-002 (PCA rolling W={GMM_WINDOW})":     viz_gmm_val,
            f"IF-013  (x_f8 diff + rolling W={IF_WINDOW})": viz_if_val,
            "Ensemble (rank 평균)":                        viz_ens_val,
        },
        val_labels,
        save_path=str(OUTPUT_DIR / "005_score_trace.png"),
    )

    plot_full(viz_ens_val, val_labels, viz_ens_test, test_labels,
              tag="005_ensemble_lof005_gmm002_if013",
              save_path=str(OUTPUT_DIR / "005_val_score_trace.png"))
    plot_zooms(viz_ens_val, val_labels, viz_ens_test, test_labels,
               tag="005_ensemble_lof005_gmm002_if013",
               save_path=str(OUTPUT_DIR / "005_val_score_zoom.png"))
    plot_score_hist(viz_ens_val, val_labels, viz_ens_test, test_labels,
                    tag="005_ensemble_lof005_gmm002_if013",
                    save_path=str(OUTPUT_DIR / "005_score_hist.png"))

    print("\n=== 완료 ===")
