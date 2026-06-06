# 다변량 시계열 이상탐지 과제

비지도 학습 기반으로 다변량 시계열 데이터에서 이상 구간(Point / Contextual / Collective)을 탐지합니다.
Isolation Forest, GMM, One-Class SVM 등 전통 ML 모델을 실험하고, 앙상블로 최종 성능을 끌어올립니다.
평가 지표는 timestep 단위 **AUROC** 와 **AUPR** 이며, 최종 결과물은 `assignment_submission.csv` 입니다.

---

## 디렉토리 구조

```
ml-assignment/
├── data/
│   ├── train.csv                   # 학습용 (정상만)
│   ├── val.csv                     # 검증용 (라벨 포함)
│   ├── test_public.csv             # 자체 평가용 (라벨 포함)
│   └── test_hidden_no_labels.csv   # 최종 제출용 (라벨 없음)
│
├── src/                            # 공통 모듈 (실험 간 공유)
│   ├── __init__.py
│   ├── data_loader.py              # CSV 로드, label 분리
│   ├── preprocessing.py            # 결측치 처리, 스케일링, Strategy A/B 분기
│   ├── features.py                 # 슬라이딩 윈도우 + 통계 피처 추출
│   ├── models.py                   # IF / GMM / OCSVM 학습 헬퍼
│   ├── ensemble.py                 # Rank 정규화, 앙상블, AUPR/AUROC 평가
│   └── evaluate.py                 # 시각화 (plot_full, plot_zooms)
│
├── experiments/                    # 실험 스크립트 (exp01 → exp06 순서로 실행)
│   ├── exp01_if_baseline.py        # 실험 01: Isolation Forest Baseline
│   ├── exp02_gmm_baseline.py       # 실험 02: GMM Baseline
│   ├── exp03_ocsvm_baseline.py     # 실험 03: One-Class SVM Baseline
│   ├── exp04_pca_effect.py         # 실험 04: PCA 전처리 효과 검증
│   ├── exp05_window_search.py      # 실험 05: 윈도우 크기 탐색
│   ├── exp06_ensemble.py           # 실험 06: 다중 모델 앙상블
│   └── logs/                       # 실험 결과 JSON 로그 (gitignore)
│
├── eda/
│   ├── eda.py                      # EDA 분석 스크립트
│   └── outputs/                    # EDA 출력 이미지 (gitignore)
│
├── results/
│   └── experiment_summary.csv      # 전체 실험 결과 요약표
│
├── main.py                         # 최종 제출 스크립트 (추론 → submission.csv 생성)
├── starter.py                      # 과제 제공 기본 코드 (수정 금지)
├── requirements.txt
└── README.md
```

---

## 실험 실행 순서

```bash
# 1. 베이스라인 확립
python experiments/exp01_if_baseline.py    # Isolation Forest 성능 측정
python experiments/exp02_gmm_baseline.py   # GMM 성능 측정
python experiments/exp03_ocsvm_baseline.py # One-Class SVM 성능 측정

# 2. 피처/전처리 개선
python experiments/exp04_pca_effect.py    # Strategy A vs B 비교

# 3. 하이퍼파라미터 탐색
python experiments/exp05_window_search.py  # 윈도우 크기 민감도 분석

# 4. 최종 앙상블
python experiments/exp06_ensemble.py       # 최적 모델 조합 앙상블

# 5. 최종 제출 파일 생성
# → results/experiment_summary.csv 보고 main.py 파라미터 채운 후 실행
python main.py                             # → assignment_submission.csv
```

---

## 데이터 설명

| 파일 | 라벨 | 용도 |
|---|---|---|
| `train.csv` | 없음 (모두 정상) | 모델 학습 |
| `val.csv` | 있음 (0=정상, 1=이상) | 하이퍼파라미터 튜닝 |
| `test_public.csv` | 있음 | 자체 성능 검증 |
| `test_hidden_no_labels.csv` | 없음 | 최종 제출 (강사 채점) |

- 컬럼: `t` (timestep) + `x_xx` 형식 채널 10개 + `label` (일부 파일만)
- 일부 채널은 연속형, 일부는 이산형 (EDA로 파악)

## 평가

- timestep 단위 **AUROC**, **AUPR** 두 지표로 채점
- `score`는 **클수록 이상**으로 간주
- 제출 형식: `t, score` 두 컬럼의 CSV
