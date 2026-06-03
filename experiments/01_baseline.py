# experiments/01_baseline.py
"""
Exp 1 — Baseline
=================================================================
목적:
  비지도 이상탐지 세 모델의 초기 성능을 측정하고,
  이후 실험(Exp 2: PCA 추가, Exp 3: window 튜닝)의 비교 기준점을 확립한다.

  모델별 조건:
    IF   : 전략 A (10채널, 스케일링 없음), window=50
           contamination ∈ {0.001, 0.005, 0.01}
    OCSVM: 전략 B (7채널, StandardScaler, PCA 없음), window=200
           nu ∈ {0.01, 0.03, 0.05, 0.1}
    GMM  : 전략 B (7채널, StandardScaler, PCA 없음), window=200
           n_components ∈ {2, 3, 5, 8}

  ※ PCA는 Exp 2에서 추가한다. 이 실험은 그 비교 기준이 된다.
=================================================================
"""

import os
import sys
import time
import warnings

# src/, starter.py 를 import 하기 위해 저장소 루트를 경로에 추가
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서도 PNG 저장 가능
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from starter import load_split
from src.features import extract_statistical_features
from src.models import fit_isolation_forest, fit_ocsvm, fit_gmm
from src.ensemble import to_anomaly_score, rank_normalize, evaluate_auroc, evaluate_aupr

# ── 전역 상수 ────────────────────────────────────────────────────────────────
DISCRETE_COLS = ['x_06', 'x_92', 'x_4b']
DATA_DIR      = os.path.join(REPO_ROOT, 'data')
RESULTS_DIR   = os.path.join(REPO_ROOT, 'results')
MODELS_DIR    = os.path.join(RESULTS_DIR, 'models')

# 이상 유형 구간 길이 정의 (양 끝 포함)
ANOMALY_TYPE_RANGES = {
    'point':      (1,   5),      # 1~5 step: Point anomaly
    'contextual': (6,   200),    # 6~200 step: Contextual anomaly
    'collective': (201, int(1e9)),  # 201+ step: Collective anomaly
}


# ── 라벨 정렬 ─────────────────────────────────────────────────────────────────
def make_window_labels(labels: np.ndarray, window_size: int) -> np.ndarray:
    """
    rolling window 끝 timestep 기준 라벨을 생성한다.

    extract_statistical_features는 rolling(min_periods=1)을 사용하므로
    출력 행 수 = 입력 행 수 = T 를 유지한다.
    위치 i의 피처는 window [i-w+1, i]를 요약하므로,
    라벨은 window 마지막 timestep i의 원래 라벨을 그대로 사용한다.

    Parameters
    ----------
    labels : np.ndarray, shape (T,)
    window_size : int  (1 이상이어야 함, 호출부 실수 조기 감지용)

    Returns
    -------
    np.ndarray, shape (T,)
    """
    if window_size < 1:
        raise ValueError(f"window_size는 1 이상이어야 합니다. 입력값: {window_size}")
    return labels.copy()


# ── 전처리 ───────────────────────────────────────────────────────────────────
def preprocess_strategy_a(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, None]:
    """
    전략 A: 10채널 전체를 스케일링 없이 반환한다.

    IsolationForest는 분리 경로 길이를 사용하므로 스케일 불변.
    """
    return (
        train_df[feature_cols].copy(),
        val_df[feature_cols].copy(),
        test_df[feature_cols].copy(),
        None,
    )


def preprocess_strategy_b_no_pca(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    전략 B (이번 실험 한정 — PCA 없음):
      이산형 3채널 제거 → 연속형 7채널만 StandardScaler 적용.

    train으로만 fit_transform, val/test는 transform만 (Data Leakage 방지).
    PCA는 Exp 2에서 추가하며, 이 실험은 PCA 유무의 비교 기준이 된다.
    """
    cont_cols = [c for c in feature_cols if c not in DISCRETE_COLS]

    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(train_df[cont_cols].values)
    Xvl    = scaler.transform(val_df[cont_cols].values)
    Xte    = scaler.transform(test_df[cont_cols].values)

    def _to_df(X: np.ndarray, ref: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X, columns=cont_cols, index=ref.index)

    return _to_df(Xtr, train_df), _to_df(Xvl, val_df), _to_df(Xte, test_df), scaler


# ── 하이퍼파라미터 탐색 ──────────────────────────────────────────────────────
def search_best_params(
    model_type: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    val_labels: np.ndarray,
    param_grid: list[dict],
) -> tuple[dict, float, np.ndarray, object]:
    """
    param_grid 내 각 조합으로 모델을 학습하고 val AUPR 기준 최적 조합을 반환한다.

    탐색 중 각 조합의 AUPR을 실시간 출력한다.

    Parameters
    ----------
    model_type : str  ('if', 'ocsvm', 'gmm')
    param_grid : list[dict]  각 dict가 하나의 하이퍼파라미터 조합

    Returns
    -------
    best_params : dict
    best_aupr   : float
    best_r_val  : np.ndarray  rank-normalized val anomaly score (클수록 이상)
    best_model  : 학습된 모델 객체  (test 스코어링에 재사용)
    """
    fit_fn = {
        'if':    fit_isolation_forest,
        'ocsvm': fit_ocsvm,
        'gmm':   fit_gmm,
    }[model_type]

    best_aupr, best_params, best_r_val, best_model = -1.0, None, None, None

    for params in param_grid:
        model, _, s_val = fit_fn(X_train, X_val, **params)
        r_val = rank_normalize(to_anomaly_score(s_val, model_type))
        aupr  = evaluate_aupr(r_val, val_labels)
        print(f"    [{model_type.upper():5s}] {str(params):35s} → val AUPR = {aupr:.4f}")

        if aupr > best_aupr:
            best_aupr   = aupr
            best_params = params
            best_r_val  = r_val
            best_model  = model

    return best_params, best_aupr, best_r_val, best_model


# ── 이상 유형별 AUPR 분해 ────────────────────────────────────────────────────
def _find_anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """연속 이상 구간을 (start, end) 리스트로 반환한다. end는 미포함 인덱스."""
    segments: list[tuple[int, int]] = []
    in_anom, start = False, None
    for i, l in enumerate(labels):
        if l == 1 and not in_anom:
            in_anom, start = True, i
        elif l == 0 and in_anom:
            segments.append((start, i))
            in_anom = False
    if in_anom:
        segments.append((start, len(labels)))
    return segments


def anomaly_type_aupr(
    scores: np.ndarray,
    labels: np.ndarray,
    min_len: int,
    max_len: int,
) -> float | None:
    """
    특정 길이 범위에 해당하는 이상 구간에 대한 부분 AUPR을 계산한다.

    해당 유형 구간 timestep + 전체 정상 timestep만 추려 평가한다.
    해당 유형 구간이 없으면 None을 반환한다.

    Parameters
    ----------
    scores          : rank-normalized anomaly score (클수록 이상)
    labels          : 0=정상, 1=이상
    min_len, max_len: 포함할 이상 구간 길이 범위 (양 끝 포함)
    """
    segments   = _find_anomaly_segments(labels)
    target_idx = [
        i
        for s, e in segments
        if min_len <= (e - s) <= max_len
        for i in range(s, e)
    ]

    if not target_idx:
        return None

    normal_idx = np.where(labels == 0)[0].tolist()
    all_idx    = np.array(sorted(set(target_idx + normal_idx)))

    return evaluate_aupr(scores[all_idx], labels[all_idx])


# ── 시각화 ───────────────────────────────────────────────────────────────────
def _shade_anomalies(ax: plt.Axes, labels: np.ndarray) -> None:
    """axes에 실제 이상 구간을 빨간 배경으로 표시한다."""
    for s, e in _find_anomaly_segments(labels):
        ax.axvspan(s, e, color='red', alpha=0.15, linewidth=0)


def plot_scores(
    val_df: pd.DataFrame,
    val_labels: np.ndarray,
    score_dict: dict[str, np.ndarray],
    save_path: str,
) -> None:
    """
    val 시계열(x_3a) + 모델별 anomaly score + 이상 구간(빨간 배경) 시각화.

    Parameters
    ----------
    val_df     : load_split('val') 에서 반환된 DataFrame
    val_labels : 정답 레이블 (0=정상, 1=이상)
    score_dict : {'모델명': rank-normalized score}
    save_path  : PNG 저장 경로
    """
    T = len(val_labels)
    t = np.arange(T)
    colors = {'IF': '#2ca02c', 'OCSVM': '#ff7f0e', 'GMM': '#9467bd'}

    n_rows = 1 + len(score_dict)
    fig, axes = plt.subplots(n_rows, 1, figsize=(22, 3 * n_rows), sharex=True)

    # 대표 채널 신호
    axes[0].plot(t, val_df['x_3a'].values, color='steelblue', linewidth=0.5)
    axes[0].set_ylabel('x_3a', fontsize=9)
    axes[0].set_title(
        'Validation Set — Signal & Anomaly Scores  (빨간 배경 = 실제 이상 구간)',
        fontsize=11,
    )
    _shade_anomalies(axes[0], val_labels)

    # 모델별 anomaly score
    for ax, (name, scores) in zip(axes[1:], score_dict.items()):
        ax.plot(t, scores, color=colors.get(name, 'gray'), linewidth=0.6, label=name)
        ax.set_ylabel('Score (0~1)', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right', fontsize=9)
        _shade_anomalies(ax, val_labels)

    axes[-1].set_xlabel('Timestep', fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → 저장: {save_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    t_start = time.time()
    warnings.filterwarnings('ignore', category=UserWarning)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    SEP = '=' * 65

    # ── [1] 데이터 로드 ────────────────────────────────────────────────────
    print(SEP)
    print('[1] 데이터 로드')
    print(SEP)
    train_df, feature_cols, _           = load_split('train',       data_dir=DATA_DIR)
    val_df,   _,            val_labels  = load_split('val',         data_dir=DATA_DIR)
    test_df,  _,            test_labels = load_split('test_public', data_dir=DATA_DIR)

    print(f'  train : {train_df.shape}  (정상만, label 없음)')
    print(f'  val   : {val_df.shape},  이상 비율 = {val_labels.mean()*100:.1f}%')
    print(f'  test  : {test_df.shape},  이상 비율 = {test_labels.mean()*100:.1f}%')
    print(f'  feature_cols ({len(feature_cols)}개): {feature_cols}')

    # ── [2] 전처리 ─────────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('[2] 전처리')
    print(SEP)
    # 전략 A: 10채널 전체, 스케일링 없음 (IF용)
    tr_A, vl_A, te_A, _      = preprocess_strategy_a(
        train_df, val_df, test_df, feature_cols
    )
    # 전략 B (PCA 없음): 7채널 + StandardScaler (OCSVM/GMM용)
    tr_B, vl_B, te_B, scaler = preprocess_strategy_b_no_pca(
        train_df, val_df, test_df, feature_cols
    )
    print(f'  전략A (10ch, 스케일링 없음): train={tr_A.shape}')
    print(f'  전략B (7ch, StandardScaler, PCA 없음): train={tr_B.shape}')

    # ── [3] 통계 피처 추출 ─────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('[3] 통계 피처 추출 (rolling window)')
    print(SEP)
    # IF: window=50, 전략A (10ch x 6stats = 60, 이산형 포함시 +3 ratio = 63 or 45)
    feat_tr_A = extract_statistical_features(tr_A, window_size=50,  strategy='A')
    feat_vl_A = extract_statistical_features(vl_A, window_size=50,  strategy='A')
    feat_te_A = extract_statistical_features(te_A, window_size=50,  strategy='A')
    # OCSVM/GMM: window=200, 연속형 7ch x 6stats = 42 features
    # (x_ 컬럼이지만 이산형 없으므로 strategy='A'로 연속형만 처리)
    feat_tr_B = extract_statistical_features(tr_B, window_size=200, strategy='A')
    feat_vl_B = extract_statistical_features(vl_B, window_size=200, strategy='A')
    feat_te_B = extract_statistical_features(te_B, window_size=200, strategy='A')

    print(f'  IF용 feat_A : {feat_tr_A.shape}  (window=50)')
    print(f'  OCSVM/GMM용 feat_B : {feat_tr_B.shape}  (window=200)')

    X_tr_A, X_vl_A, X_te_A = feat_tr_A.values, feat_vl_A.values, feat_te_A.values
    X_tr_B, X_vl_B, X_te_B = feat_tr_B.values, feat_vl_B.values, feat_te_B.values

    # ── [4] 라벨 정렬 ──────────────────────────────────────────────────────
    # rolling stats은 min_periods=1로 T행 유지 → 라벨과 1:1 대응
    vl_lbl_A  = make_window_labels(val_labels,  window_size=50)
    vl_lbl_B  = make_window_labels(val_labels,  window_size=200)
    te_lbl_A  = make_window_labels(test_labels, window_size=50)
    te_lbl_B  = make_window_labels(test_labels, window_size=200)

    assert len(X_vl_A) == len(vl_lbl_A),  'IF val: 피처-라벨 행 수 불일치'
    assert len(X_vl_B) == len(vl_lbl_B),  'OCSVM/GMM val: 피처-라벨 행 수 불일치'
    assert len(X_te_A) == len(te_lbl_A),  'IF test: 피처-라벨 행 수 불일치'
    assert len(X_te_B) == len(te_lbl_B),  'OCSVM/GMM test: 피처-라벨 행 수 불일치'

    # ── [5] 하이퍼파라미터 탐색 ────────────────────────────────────────────
    print(f'\n{SEP}')
    print('[5] 하이퍼파라미터 탐색 (val AUPR 기준)')
    print(SEP)

    print('  IF — contamination 탐색')
    best_if_p, _, r_vl_if, model_if = search_best_params(
        'if', X_tr_A, X_vl_A, vl_lbl_A,
        [{'contamination': c} for c in [0.001, 0.005, 0.01]],
    )

    print('  OCSVM — nu 탐색')
    best_oc_p, _, r_vl_oc, model_oc = search_best_params(
        'ocsvm', X_tr_B, X_vl_B, vl_lbl_B,
        [{'nu': n} for n in [0.01, 0.03, 0.05, 0.1]],
    )

    print('  GMM — n_components 탐색')
    best_gm_p, _, r_vl_gm, model_gm = search_best_params(
        'gmm', X_tr_B, X_vl_B, vl_lbl_B,
        [{'n_components': k} for k in [2, 3, 5, 8]],
    )

    print(f'\n  최적 파라미터 요약:')
    print(f'    IF    → {best_if_p}')
    print(f'    OCSVM → {best_oc_p}')
    print(f'    GMM   → {best_gm_p}')

    # ── [6] Test score 계산 ────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('[6] Test score 계산 (학습된 모델 재사용)')
    print(SEP)
    # 탐색에서 반환한 best_model로 바로 채점 (재학습 불필요)
    r_te_if = rank_normalize(
        to_anomaly_score(model_if.score_samples(X_te_A), 'if')
    )
    r_te_oc = rank_normalize(
        to_anomaly_score(model_oc.decision_function(X_te_B), 'ocsvm')
    )
    r_te_gm = rank_normalize(
        to_anomaly_score(model_gm.score_samples(X_te_B), 'gmm')
    )
    print('  완료')

    # ── [7] 평가 ───────────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('[7] 평가')
    print(SEP)

    model_specs = [
        # (이름, 전략, window, val라벨, test라벨, val score, test score, params)
        ('IF',    'A', 50,  vl_lbl_A, te_lbl_A, r_vl_if, r_te_if, best_if_p),
        ('OCSVM', 'B', 200, vl_lbl_B, te_lbl_B, r_vl_oc, r_te_oc, best_oc_p),
        ('GMM',   'B', 200, vl_lbl_B, te_lbl_B, r_vl_gm, r_te_gm, best_gm_p),
    ]

    results = []
    for model_name, strat, ws, vl_lbl, te_lbl, r_val, r_te, params in model_specs:
        val_auroc = evaluate_auroc(r_val, vl_lbl)
        val_aupr  = evaluate_aupr(r_val,  vl_lbl)
        te_auroc  = evaluate_auroc(r_te,  te_lbl)
        te_aupr   = evaluate_aupr(r_te,   te_lbl)

        pt = anomaly_type_aupr(r_val, vl_lbl, *ANOMALY_TYPE_RANGES['point'])
        ct = anomaly_type_aupr(r_val, vl_lbl, *ANOMALY_TYPE_RANGES['contextual'])
        cl = anomaly_type_aupr(r_val, vl_lbl, *ANOMALY_TYPE_RANGES['collective'])

        fmt = lambda x: f'{x:.4f}' if x is not None else '  N/A '
        print(
            f'  {model_name:5s} | '
            f'val AUROC={val_auroc:.4f} AUPR={val_aupr:.4f} | '
            f'Point={fmt(pt)} Ctx={fmt(ct)} Col={fmt(cl)} | '
            f'test AUROC={te_auroc:.4f} AUPR={te_aupr:.4f}'
        )

        results.append({
            'exp':             '01_baseline',
            'model':           model_name,
            'strategy':        strat,
            'window_size':     ws,
            'best_params':     str(params),
            'val_auroc':       round(val_auroc, 4),
            'val_aupr':        round(val_aupr,  4),
            'point_aupr':      round(pt, 4) if pt is not None else None,
            'contextual_aupr': round(ct, 4) if ct is not None else None,
            'collective_aupr': round(cl, 4) if cl is not None else None,
            'test_auroc':      round(te_auroc, 4),
            'test_aupr':       round(te_aupr,  4),
        })

    # ── [8] 결과 저장 ──────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('[8] 결과 저장')
    print(SEP)
    results_df = pd.DataFrame(results)
    csv_path   = os.path.join(RESULTS_DIR, '01_baseline_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'  결과 CSV → {csv_path}')
    print()
    print(results_df.to_string(index=False))

    # ── [9] 시각화 ─────────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('[9] 시각화')
    print(SEP)
    plot_scores(
        val_df,
        val_labels,
        score_dict={'IF': r_vl_if, 'OCSVM': r_vl_oc, 'GMM': r_vl_gm},
        save_path=os.path.join(RESULTS_DIR, '01_baseline_score_plot.png'),
    )

    # ── [10] 모델 저장 ─────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('[10] 모델 저장 (Exp 2에서 동일 조건 비교용)')
    print(SEP)
    joblib.dump(model_if, os.path.join(MODELS_DIR, 'baseline_if.pkl'))
    joblib.dump(model_oc, os.path.join(MODELS_DIR, 'baseline_ocsvm.pkl'))
    joblib.dump(model_gm, os.path.join(MODELS_DIR, 'baseline_gmm.pkl'))
    joblib.dump(scaler,   os.path.join(MODELS_DIR, 'baseline_scaler_B.pkl'))
    print(f'  baseline_if.pkl        → IsolationForest {best_if_p}')
    print(f'  baseline_ocsvm.pkl     → OneClassSVM {best_oc_p}')
    print(f'  baseline_gmm.pkl       → GaussianMixture {best_gm_p}')
    print(f'  baseline_scaler_B.pkl  → StandardScaler (7채널, train fit)')

    print(f'\n{SEP}')
    print(f'완료 — 총 소요 시간: {time.time() - t_start:.1f}초')
    print(SEP)


if __name__ == '__main__':
    main()
