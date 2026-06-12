"""앙상블 001: IF + LOF — Rank 평균 앙상블

IF: x_f8 차분 → 연속형 Z-Score Rolling + 이산형 Rolling Mean (IF-010 기반)
LOF: 연속형 Rolling 통계 + StandardScaler (LOF-009 기반)
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
from scipy.stats import pearsonr, spearmanr

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader   import load_data
from src.preprocessing import fill_missing, fit_scaler, apply_scaler
from src.features      import DISCRETE_COLS, rolling_features
from src.models        import fit_isolation_forest, fit_lof
from src.ensemble      import flip_score, rank_normalize, ensemble_scores
from src.evaluate      import (evaluate_aupr, evaluate_auroc,
                               anomaly_type_aupr, plot_full, plot_zooms,
                               plot_score_hist, _find_segments, _draw_full)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "experiments" / "ensemble" / "outputs"

# ── IF 하이퍼파라미터 ────────────────────────────────────────────
IF_WINDOW      = 50
IF_ESTIMATORS  = 300
IF_CONTAM      = 0.0001
IF_RANDOM      = 42
IF_DIFF_COL    = "x_f8"

# ── LOF 하이퍼파라미터 ───────────────────────────────────────────
LOF_WINDOW     = 5
LOF_NEIGHBORS  = 10
LOF_CONTAM     = 0.0001
LOF_STATS      = ["mean", "std", "min", "max", "range"]

POINT_LEN      = (1,   5)
CONTEXTUAL_LEN = (6,   200)
COLLECTIVE_LEN = (201, 10**9)

# 앙상블 조합은 rank_normalize 고정.
# VIZ_MINMAX=True: 시각화 직전에만 minmax 재정규화 (AUPR/AUROC 불변)
VIZ_MINMAX = True


def minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


# ── IF 피처 빌더 ─────────────────────────────────────────────────

def _apply_diff(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].diff(periods=1).fillna(0)
    return df


def _zscore_rolling(df: pd.DataFrame, cols: list[str], w: int) -> pd.DataFrame:
    """look-ahead 없는 z-score: (현재 − shift(1) rolling mean) / (shift(1) rolling std + ε)"""
    past   = df[cols].shift(1)
    r_mean = past.rolling(w, min_periods=1).mean()
    r_std  = past.rolling(w, min_periods=1).std().fillna(0)
    z      = (df[cols] - r_mean) / (r_std + 1e-8)
    z.columns = [f"{c}_z" for c in cols]
    return z.fillna(0)


def build_if_features(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    cont_cols: list[str],
    disc_cols:  list[str],
    w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr_z = _zscore_rolling(train_df, cont_cols, w)
    va_z = _zscore_rolling(val_df,   cont_cols, w)
    te_z = _zscore_rolling(test_df,  cont_cols, w)

    tr_d = train_df[disc_cols].rolling(w, min_periods=1).mean().fillna(0)
    va_d = val_df[disc_cols].rolling(w, min_periods=1).mean().fillna(0)
    te_d = test_df[disc_cols].rolling(w, min_periods=1).mean().fillna(0)

    return (
        np.hstack([tr_z.to_numpy(), tr_d.to_numpy()]),
        np.hstack([va_z.to_numpy(), va_d.to_numpy()]),
        np.hstack([te_z.to_numpy(), te_d.to_numpy()]),
    )


# ── LOF 피처 빌더 ────────────────────────────────────────────────

def build_lof_features(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    cont_cols: list[str],
    w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr_feat = rolling_features(train_df, cols=cont_cols, window_size=w, stats=LOF_STATS)
    va_feat = rolling_features(val_df,   cols=cont_cols, window_size=w, stats=LOF_STATS)
    te_feat = rolling_features(test_df,  cols=cont_cols, window_size=w, stats=LOF_STATS)

    scaler   = fit_scaler(tr_feat)
    tr_scaled = apply_scaler(scaler, tr_feat)
    va_scaled = apply_scaler(scaler, va_feat)
    te_scaled = apply_scaler(scaler, te_feat)

    return (
        tr_scaled.to_numpy(),
        va_scaled.to_numpy(),
        te_scaled.to_numpy(),
    )


# ── 다양성 분석 시각화 ───────────────────────────────────────────

def plot_diversity(
    if_val:      np.ndarray,
    lof_val:     np.ndarray,
    val_labels:  np.ndarray,
    if_test:     np.ndarray,
    lof_test:    np.ndarray,
    test_labels: np.ndarray,
    ens_val:     np.ndarray,
    ens_test:    np.ndarray,
    save_path:   str,
    viz_if_val:  np.ndarray | None = None,
    viz_lof_val: np.ndarray | None = None,
    viz_ens_val: np.ndarray | None = None,
) -> None:
    """4패널 다양성 진단 플롯."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("앙상블 001 — 모델 다양성 진단", fontsize=13)

    # ── 패널 [0,0]: Score 산점도 (IF vs LOF, val) ─────────────
    ax = axes[0, 0]
    normal_mask  = val_labels == 0
    anomaly_mask = val_labels == 1

    sample = np.random.default_rng(0).choice(
        normal_mask.sum(), size=min(5000, normal_mask.sum()), replace=False
    )
    ax.scatter(if_val[normal_mask][sample],  lof_val[normal_mask][sample],
               c="steelblue", s=4, alpha=0.3, label="Val 정상")
    ax.scatter(if_val[anomaly_mask], lof_val[anomaly_mask],
               c="tomato", s=16, alpha=0.85, label="Val 이상")

    # 대각선 (두 모델이 완전히 일치하는 선)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="y=x (완전 일치)")

    pcc_val, _ = pearsonr(if_val,   lof_val)
    scc_val, _ = spearmanr(if_val,  lof_val)
    pcc_test, _ = pearsonr(if_test, lof_test)
    scc_test, _ = spearmanr(if_test, lof_test)

    ax.set_xlabel("IF Rank Score")
    ax.set_ylabel("LOF Rank Score")
    ax.set_title(
        f"Score 산점도 (val)\n"
        f"Pearson r = {pcc_val:.3f}  |  Spearman ρ = {scc_val:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=8, markerscale=3)

    # ── 패널 [0,1]: 상관계수 히트맵 ─────────────────────────
    ax = axes[0, 1]
    corr_data = np.array([
        [1.0,     pcc_val,  pearsonr(if_val,  ens_val)[0]],
        [pcc_val, 1.0,      pearsonr(lof_val, ens_val)[0]],
        [pearsonr(if_val, ens_val)[0], pearsonr(lof_val, ens_val)[0], 1.0],
    ])
    im = ax.imshow(corr_data, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)

    labels_heat = ["IF", "LOF", "Ensemble"]
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels_heat)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(labels_heat)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr_data[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, color="black")
    ax.set_title(
        f"Pearson 상관계수 히트맵 (val)\n"
        f"val: IF↔LOF r={pcc_val:.3f}, ρ={scc_val:.3f}  |  "
        f"test: IF↔LOF r={pcc_test:.3f}, ρ={scc_test:.3f}",
        fontsize=9,
    )

    # ── 패널 [1,0]: 유형별 AUPR 막대 비교 ───────────────────
    ax = axes[1, 0]
    types = ["Overall", "Point\n(1~5)", "Contextual\n(6~200)", "Collective\n(201+)"]

    def _type_aupr(scores, labels):
        return [
            evaluate_aupr(scores, labels),
            anomaly_type_aupr(scores, labels, *POINT_LEN),
            anomaly_type_aupr(scores, labels, *CONTEXTUAL_LEN),
            anomaly_type_aupr(scores, labels, *COLLECTIVE_LEN),
        ]

    if_auprs  = _type_aupr(if_val,  val_labels)
    lof_auprs = _type_aupr(lof_val, val_labels)
    ens_auprs = _type_aupr(ens_val, val_labels)

    x = np.arange(len(types))
    w = 0.25
    bars_if  = ax.bar(x - w,   if_auprs,  w, label="IF",       color="steelblue", alpha=0.8)
    bars_lof = ax.bar(x,       lof_auprs, w, label="LOF",      color="orange",    alpha=0.8)
    bars_ens = ax.bar(x + w,   ens_auprs, w, label="Ensemble", color="seagreen",  alpha=0.8)

    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    _label_bars(bars_if)
    _label_bars(bars_lof)
    _label_bars(bars_ens)

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=9)
    ax.set_ylabel("AUPR")
    ax.set_ylim(0, min(1.2, ax.get_ylim()[1] * 1.25))
    ax.set_title("Val 유형별 AUPR 비교 (IF / LOF / Ensemble)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── 패널 [1,1]: 이상 구간 score KDE (val) ───────────────
    ax = axes[1, 1]
    from scipy.stats import gaussian_kde

    _kde_if  = viz_if_val  if viz_if_val  is not None else if_val
    _kde_lof = viz_lof_val if viz_lof_val is not None else lof_val
    _kde_ens = viz_ens_val if viz_ens_val is not None else ens_val
    for scores, name, color in [
        (_kde_if,  "IF",       "steelblue"),
        (_kde_lof, "LOF",      "orange"),
        (_kde_ens, "Ensemble", "seagreen"),
    ]:
        normal_s  = scores[val_labels == 0]
        anomaly_s = scores[val_labels == 1]
        xs = np.linspace(0, 1, 300)
        try:
            ax.plot(xs, gaussian_kde(normal_s)(xs),  lw=1.5, ls="--",
                    color=color, alpha=0.6, label=f"{name} 정상")
            ax.plot(xs, gaussian_kde(anomaly_s)(xs), lw=2.0, ls="-",
                    color=color, alpha=0.9, label=f"{name} 이상")
        except Exception:
            pass

    score_label = "Minmax Score" if VIZ_MINMAX else "Rank Score"
    ax.set_xlabel(score_label)
    ax.set_ylabel("KDE 밀도")
    ax.set_title(f"Score KDE ({score_label}): 정상(점선) vs 이상(실선) — val", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")


def plot_score_compare(
    scores_dict: dict[str, np.ndarray],
    val_labels: np.ndarray,
    save_path: str,
) -> None:
    """IF / LOF / Ensemble score를 3행으로 나란히 비교 (val 기준)."""
    n = len(scores_dict)
    fig, axes = plt.subplots(n, 1, figsize=(18, 4 * n))
    if n == 1:
        axes = [axes]
    for ax, (name, scores) in zip(axes, scores_dict.items()):
        aupr  = evaluate_aupr(scores, val_labels)
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
    print("=== 앙상블 001: IF + LOF (Rank 평균) ===\n")

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

    # ── IF ────────────────────────────────────────────────────────
    print("\n[IF] 파이프라인 구성 중...")
    tr_diff = _apply_diff(train_df, IF_DIFF_COL)
    va_diff = _apply_diff(val_df,   IF_DIFF_COL)
    te_diff = _apply_diff(test_df,  IF_DIFF_COL)

    if_train_X, if_val_X, if_test_X = build_if_features(
        tr_diff, va_diff, te_diff, cont_cols, disc_cols, IF_WINDOW
    )
    print(f"  IF feature dim: {if_train_X.shape[1]}  "
          f"(cont {len(cont_cols)}×z + disc {len(disc_cols)}×ratio, W={IF_WINDOW})")

    print("  IF 학습 중...")
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

    # ── LOF ───────────────────────────────────────────────────────
    print("\n[LOF] 파이프라인 구성 중...")
    lof_train_X, lof_val_X, lof_test_X = build_lof_features(
        train_df, val_df, test_df, cont_cols, LOF_WINDOW
    )
    print(f"  LOF feature dim: {lof_train_X.shape[1]}  "
          f"(cont {len(cont_cols)} × {len(LOF_STATS)} stats, W={LOF_WINDOW})")

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

    # ── 앙상블 ────────────────────────────────────────────────────
    print("\n[Ensemble] rank 평균 앙상블...")
    ens_val_scores  = ensemble_scores({"if": if_val_scores,  "lof": lof_val_scores})
    ens_test_scores = ensemble_scores({"if": if_test_scores, "lof": lof_test_scores})

    # ── 상관계수 출력 ─────────────────────────────────────────────
    pcc_val,  _ = pearsonr(if_val_scores,   lof_val_scores)
    scc_val,  _ = spearmanr(if_val_scores,  lof_val_scores)
    pcc_test, _ = pearsonr(if_test_scores,  lof_test_scores)
    scc_test, _ = spearmanr(if_test_scores, lof_test_scores)
    print(f"\n[다양성] IF ↔ LOF score 상관계수")
    print(f"  val  : Pearson r={pcc_val:.4f}  Spearman ρ={scc_val:.4f}")
    print(f"  test : Pearson r={pcc_test:.4f}  Spearman ρ={scc_test:.4f}")
    if abs(scc_val) < 0.5:
        print("  → 상관이 낮아 앙상블 효과 기대 ✓")
    elif abs(scc_val) < 0.75:
        print("  → 중간 상관 — 앙상블 이득 보통")
    else:
        print("  → 상관이 높아 앙상블 이득 제한적")

    # ── 최종 평가 ─────────────────────────────────────────────────
    print(f"\n[최종 평가]")
    header = f"  {'모델':<12} {'val AUROC':>10} {'val AUPR':>9} {'test AUROC':>11} {'test AUPR':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, vs, vl, ts, tl in [
        ("IF",       if_val_scores,  val_labels, if_test_scores,  test_labels),
        ("LOF",      lof_val_scores, val_labels, lof_test_scores, test_labels),
        ("Ensemble", ens_val_scores, val_labels, ens_test_scores, test_labels),
    ]:
        print(f"  {name:<12} {evaluate_auroc(vs, vl):>10.4f} {evaluate_aupr(vs, vl):>9.4f}"
              f" {evaluate_auroc(ts, tl):>11.4f} {evaluate_aupr(ts, tl):>9.4f}")

    print(f"\n  [Val] 유형별 AUPR")
    print(f"  {'유형':<14} {'IF':>8} {'LOF':>8} {'Ensemble':>10}")
    print("  " + "-" * 42)
    for type_name, lo, hi in [
        ("Point(1~5)", *POINT_LEN),
        ("Contextual", *CONTEXTUAL_LEN),
        ("Collective", *COLLECTIVE_LEN),
    ]:
        ifa = anomaly_type_aupr(if_val_scores,  val_labels, lo, hi)
        lfa = anomaly_type_aupr(lof_val_scores, val_labels, lo, hi)
        ensa = anomaly_type_aupr(ens_val_scores, val_labels, lo, hi)
        def _fmt(v): return f"{v:.4f}" if not np.isnan(v) else "  nan "
        print(f"  {type_name:<14} {_fmt(ifa):>8} {_fmt(lfa):>8} {_fmt(ensa):>10}")

    print(f"\n  [Test] 유형별 AUPR")
    print(f"  {'유형':<14} {'IF':>8} {'LOF':>8} {'Ensemble':>10}")
    print("  " + "-" * 42)
    for type_name, lo, hi in [
        ("Point(1~5)", *POINT_LEN),
        ("Contextual", *CONTEXTUAL_LEN),
        ("Collective", *COLLECTIVE_LEN),
    ]:
        ifa  = anomaly_type_aupr(if_test_scores,  test_labels, lo, hi)
        lfa  = anomaly_type_aupr(lof_test_scores, test_labels, lo, hi)
        ensa = anomaly_type_aupr(ens_test_scores, test_labels, lo, hi)
        def _fmt(v): return f"{v:.4f}" if not np.isnan(v) else "   nan"
        print(f"  {type_name:<14} {_fmt(ifa):>8} {_fmt(lfa):>8} {_fmt(ensa):>10}")

    # ── 시각화용 점수 변환 ────────────────────────────────────────
    # 앙상블 조합은 rank 점수로 완료. 시각화 직전에만 minmax 재정규화.
    # AUPR/AUROC는 rank-based 지표이므로 변환 전후 동일.
    if VIZ_MINMAX:
        # 개별 모델: raw flip score에 minmax → 실제 점수 분포 형태가 시각화에 드러남
        # 앙상블: 각 모델 minmax(raw) 합산 후 최종 minmax → rank 평균보다 spike 구조 보존
        viz_if_val   = minmax(raw_if_val)
        viz_lof_val  = minmax(raw_lof_val)
        viz_ens_val  = minmax(minmax(raw_if_val)  + minmax(raw_lof_val))
        viz_ens_test = minmax(minmax(raw_if_test) + minmax(raw_lof_test))
        print("\n  [시각화] minmax 적용 — 개별: raw flip score, 앙상블: minmax 합산 후 재정규화 (평가 지표 불변)")
    else:
        viz_if_val   = if_val_scores
        viz_lof_val  = lof_val_scores
        viz_ens_val  = ens_val_scores
        viz_ens_test = ens_test_scores

    # ── 시각화 ────────────────────────────────────────────────────
    # plot_diversity 산점도·히트맵은 rank 점수 그대로 (상관 구조 보존)
    plot_diversity(
        if_val_scores, lof_val_scores, val_labels,
        if_test_scores, lof_test_scores, test_labels,
        ens_val_scores, ens_test_scores,
        save_path=str(OUTPUT_DIR / "001_diversity.png"),
        viz_if_val=viz_if_val,
        viz_lof_val=viz_lof_val,
        viz_ens_val=viz_ens_val,
    )

    plot_score_compare(
        {
            "IF  (z-score rolling W=50)": viz_if_val,
            "LOF (rolling stats)":        viz_lof_val,
            "Ensemble (rank 평균)":        viz_ens_val,
        },
        val_labels,
        save_path=str(OUTPUT_DIR / "001_score_trace.png"),
    )

    plot_full(viz_ens_val, val_labels, viz_ens_test, test_labels,
              tag="001_ensemble_if_lof",
              save_path=str(OUTPUT_DIR / "001_val_score_trace.png"))
    plot_zooms(viz_ens_val, val_labels, viz_ens_test, test_labels,
               tag="001_ensemble_if_lof",
               save_path=str(OUTPUT_DIR / "001_val_score_zoom.png"))
    plot_score_hist(viz_ens_val, val_labels, viz_ens_test, test_labels,
                    tag="001_ensemble_if_lof",
                    save_path=str(OUTPUT_DIR / "001_score_hist.png"))

    print("\n=== 완료 ===")
