"""
평가 및 시각화 모듈

평가 지표 계산과 결과 시각화를 담당합니다.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score


# ── 평가 함수 ────────────────────────────────────────────────────

def evaluate_aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUPR (Average Precision) 계산. scores는 클수록 이상."""
    return float(average_precision_score(labels, scores))


def evaluate_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC 계산. scores는 클수록 이상."""
    return float(roc_auc_score(labels, scores))


def anomaly_type_aupr(
    scores: np.ndarray,
    labels: np.ndarray,
    min_len: int,
    max_len: int,
) -> float:
    """
    길이가 [min_len, max_len] 범위인 이상 구간만 포함하여 AUPR을 계산합니다.
    해당 길이 범위의 이상 구간이 없으면 np.nan 반환.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    anomaly_mask = np.zeros(len(labels), dtype=bool)
    i = 0
    while i < len(labels):
        if labels[i] == 1:
            j = i + 1
            while j < len(labels) and labels[j] == 1:
                j += 1
            if min_len <= (j - i) <= max_len:
                anomaly_mask[i:j] = True
            i = j
        else:
            i += 1

    if anomaly_mask.sum() == 0:
        return float("nan")

    final_mask = anomaly_mask | (labels == 0)
    return float(average_precision_score(labels[final_mask], scores[final_mask]))


# ── 시각화 함수 ────────────────────────────────────────────────────

def _find_segments(arr):
    """연속 1 구간의 (start, end) 리스트 반환 (end는 exclusive)"""
    segments, in_seg = [], False
    for i, v in enumerate(arr):
        if v == 1 and not in_seg:
            seg_start, in_seg = i, True
        elif v == 0 and in_seg:
            segments.append((seg_start, i))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(arr)))
    return segments


def _draw_full(ax, scores, labels, title):
    """단일 ax에 전체 시계열 score + 이상 구간 음영"""
    first = True
    for s, e in _find_segments(labels):
        ax.axvspan(s, e, alpha=0.25, color="red",
                   label="이상 구간 (정답)" if first else "_nolegend_")
        first = False
    ax.plot(scores, lw=0.7, color="steelblue", label="Anomaly Score")
    ax.set_xlim(0, len(scores))
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)


def _draw_zooms_row(axes, scores, labels, split_name, context=150):
    """axes 배열(1행)에 이상 구간 확대 패널을 채웁니다."""
    segments = _find_segments(labels)
    for ax, (s, e) in zip(axes, segments):
        lo = max(0, s - context)
        hi = min(len(scores), e + context)

        first = True
        for ss, se in _find_segments(labels[lo:hi]):
            ax.axvspan(lo + ss, lo + se, alpha=0.25, color="red",
                       label="이상 구간" if first else "_nolegend_")
            first = False

        ax.plot(np.arange(lo, hi), scores[lo:hi], lw=0.9, color="steelblue")

        seg_len = e - s
        if seg_len == 1:
            anom_type = "Point"
        elif seg_len <= 200:
            anom_type = "Contextual"
        else:
            anom_type = "Collective"

        ax.set_title(f"[{split_name}] t={s}~{e-1}  len={seg_len}  [{anom_type}]", fontsize=8)
        ax.set_xlabel("Timestep", fontsize=7)

    for ax in axes[len(segments):]:
        ax.set_visible(False)


def plot_full(val_scores, val_labels, test_scores, test_labels, tag, save_path):
    """val / test 전체 시계열을 위아래 2행으로 출력"""
    fig, axes = plt.subplots(2, 1, figsize=(18, 7))
    _draw_full(axes[0], val_scores,  val_labels,  f"val  [{tag}]")
    _draw_full(axes[1], test_scores, test_labels, f"test [{tag}]")
    fig.suptitle(tag, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  저장: {save_path}")


def plot_zooms(val_scores, val_labels, test_scores, test_labels, tag, save_path, context=150):
    """val / test 이상 구간 확대를 위아래 2행으로 출력"""
    val_segs  = _find_segments(val_labels)
    test_segs = _find_segments(test_labels)

    if not val_segs and not test_segs:
        print("  (이상 구간 없음 — 확대 플롯 생략)")
        return

    n_cols = max(len(val_segs), len(test_segs))
    fig, axes = plt.subplots(2, n_cols, figsize=(max(5 * n_cols, 12), 7), squeeze=False)
    fig.suptitle(f"{tag} — 이상 구간 확대", fontsize=11)

    _draw_zooms_row(axes[0], val_scores,  val_labels,  "val",  context)
    _draw_zooms_row(axes[1], test_scores, test_labels, "test", context)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")


def plot_score_hist(val_scores, val_labels, test_scores, test_labels, tag, save_path, bins=80):
    """정상/이상 score 분포를 히스토그램으로 비교 (val · test 좌우 2열)

    분포가 겹쳐 있으면 모델이 구별을 못하는 것이고,
    분리되어 있으면 모델이 잘 동작하는 것입니다.
    """
    def _draw_hist(ax, scores, labels, split_name):
        scores  = np.asarray(scores)
        labels  = np.asarray(labels)
        normal  = scores[labels == 0]
        anomaly = scores[labels == 1]
        ax.hist(normal,  bins=bins, alpha=0.6, color="steelblue",
                label=f"정상  (n={len(normal):,})")
        ax.hist(anomaly, bins=bins, alpha=0.6, color="tomato",
                label=f"이상  (n={len(anomaly):,})")
        auroc = evaluate_auroc(scores, labels)
        aupr  = evaluate_aupr(scores, labels)
        ax.set_title(f"[{split_name}]  AUROC={auroc:.4f}  AUPR={aupr:.4f}", fontsize=10)
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{tag} — score 분포 (정상 vs 이상)", fontsize=11)
    _draw_hist(axes[0], val_scores,  val_labels,  "val")
    _draw_hist(axes[1], test_scores, test_labels, "test")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  저장: {save_path}")
