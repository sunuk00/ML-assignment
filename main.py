"""
최종 제출 스크립트

results/experiment_summary.csv 에서 최고 성능 실험의 파라미터를 확인한 뒤
아래 설정 블록을 채우고 실행하세요. → assignment_submission.csv 생성
"""

import os, sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.data_loader   import load_data
from src.preprocessing import (fill_missing, filter_continuous,
                                fit_scaler, apply_scaler, fit_pca, apply_pca)
from src.features      import rolling_features, window_features, DISCRETE_COLS
from src.models        import fit_isolation_forest, fit_ocsvm, fit_gmm
from src.ensemble      import flip_score, rank_normalize, smooth_scores

# ================================================================
# 설정 — 실험 완료 후 최적 파라미터로 채울 것
# ================================================================
STRATEGY      = None   # TODO: 'A' 또는 'B'
WINDOW_SIZE   = None   # TODO: ex) 100
MODEL_TYPE    = None   # TODO: 'if' / 'ocsvm' / 'gmm'
N_ESTIMATORS  = None   # TODO: IF 전용, ex) 200
CONTAMINATION = None   # TODO: IF 전용, ex) 0.01
N_COMPONENTS  = None   # TODO: GMM 전용, ex) 5
NU            = None   # TODO: OCSVM 전용, ex) 0.05
SMOOTH_WINDOW = None   # TODO: ex) 50
RANDOM_STATE  = 42
# ================================================================

DATA_DIR        = os.path.join(ROOT_DIR, 'data')
SUBMISSION_FILE = os.path.join(ROOT_DIR, 'assignment_submission.csv')


def main():
    print('=== 최종 제출 파일 생성 ===\n')

    # ── 1. 데이터 로드 ──────────────────────────────────────────────
    print('[1] 데이터 로드')
    # train_df, _   = load_data('train',                 DATA_DIR)
    # hidden_df, _  = load_data('test_hidden_no_labels', DATA_DIR)
    # print(f'  train: {train_df.shape}  hidden: {hidden_df.shape}')

    # ── 2. 결측치 처리 ──────────────────────────────────────────────
    print('[2] 결측치 처리')
    # train_df  = fill_missing(train_df)
    # hidden_df = fill_missing(hidden_df)

    # ── 3. 전처리 (Strategy 선택) ───────────────────────────────────
    print('[3] 전처리')
    # if STRATEGY == 'A':
    #     # Strategy A: 전체 10채널 그대로 사용
    #     train_ready  = train_df
    #     hidden_ready = hidden_df
    #
    # elif STRATEGY == 'B':
    #     # Strategy B: 연속형 채널 → StandardScaler → PCA
    #     train_cont  = filter_continuous(train_df)
    #     hidden_cont = filter_continuous(hidden_df)
    #
    #     scaler       = fit_scaler(train_cont)          # train으로만 fit
    #     train_scaled = apply_scaler(scaler, train_cont)
    #     hidden_scaled = apply_scaler(scaler, hidden_cont)  # 같은 scaler 사용
    #
    #     pca          = fit_pca(train_scaled)           # train으로만 fit
    #     train_ready  = apply_pca(pca, train_scaled)
    #     hidden_ready = apply_pca(pca, hidden_scaled)   # 같은 pca 사용

    # ── 4. 피처 추출 ────────────────────────────────────────────────
    print('[4] 슬라이딩 윈도우 피처 추출')
    # train_X  = rolling_stats(train_ready,  window_size=WINDOW_SIZE).to_numpy()
    # hidden_X = rolling_stats(hidden_ready, window_size=WINDOW_SIZE).to_numpy()
    # print(f'  피처 shape — train: {train_X.shape}  hidden: {hidden_X.shape}')

    # ── 5. 모델 학습 ────────────────────────────────────────────────
    print('[5] 모델 학습')
    # if MODEL_TYPE == 'if':
    #     model = fit_isolation_forest(train_X, n_estimators=N_ESTIMATORS,
    #                                  contamination=CONTAMINATION, random_state=RANDOM_STATE)
    # elif MODEL_TYPE == 'ocsvm':
    #     model = fit_ocsvm(train_X, nu=NU)
    # elif MODEL_TYPE == 'gmm':
    #     model = fit_gmm(train_X, n_components=N_COMPONENTS, random_state=RANDOM_STATE)

    # ── 6. Score 계산 ───────────────────────────────────────────────
    print('[6] Anomaly score 계산')
    # if MODEL_TYPE in ('if', 'gmm'):
    #     raw_scores = model.score_samples(hidden_X)   # 작을수록 이상
    # elif MODEL_TYPE == 'ocsvm':
    #     raw_scores = model.decision_function(hidden_X)  # 작을수록 이상
    #
    # scores = flip_score(raw_scores)          # 클수록 이상으로 반전
    # scores = rank_normalize(scores)          # [0, 1] 정규화
    # scores = smooth_scores(scores, SMOOTH_WINDOW)   # 이동 평균 스무딩

    # ── 7. 제출 파일 저장 ───────────────────────────────────────────
    print('[7] submission.csv 저장')
    # pd.DataFrame({'t': hidden_df['t'].values, 'score': scores}).to_csv(
    #     SUBMISSION_FILE, index=False
    # )
    # print(f'  저장 완료: {SUBMISSION_FILE}')
    # print(f'  행 수: {len(scores)}')

    print('\n=== TODO: 위 파라미터 설정 및 주석 해제 후 실행 ===')


if __name__ == '__main__':
    main()
