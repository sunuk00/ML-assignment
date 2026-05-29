```
project/
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test_public.csv
│   └── test_hidden_no_labels.csv
├── src/                          # 재사용할 핵심 코드를 모아두는 모듈 폴더
│   ├── __init__.py
│   ├── preprocessing.py          # 결측치 보간, 전략 A/B 분기, Scaler, PCA 담당
│   ├── features.py               # 슬라이딩 윈도우 및 통계 Feature 추출 담당
│   ├── models.py                 # IF, OCSVM, GMM 모델 선언 및 학습 헬퍼
│   └── ensemble.py               # Rank 정규화 및 가중평균 통합 담당
├── experiments/                  # 논문 작성 흐름과 1:1 매칭되는 실험 실행 스크립트
│   ├── 01_baseline.py            # 기본 전처리 + 기본 모델 성능 측정 (실험 1)
│   ├── 02_feature_ablation.py    # 전략 A vs 전략 B(PCA) 성능 비교 (실험 2)
│   ├── 03_window_tuning.py       # 윈도우 크기별 민감도 분석 (실험 3)
│   └── 04_final_ensemble.py      # 최종 앙상블 및 hidden 제출 파일 생성 (실험 4)
├── starter.py                    # 기존 제공된 기본 코드는 백업 또는 참고용으로 유지
└── requirements.txt
```

# 계획
1. 일단 EDA(Exploratory Data Analysis) → 간단한 모델 (예: Isolation Forest) → 평가 → 개선 순으로 진행
- 채널 분포, 시계열 패털, 채널 간 상관관계
2. 10개의 채널 중 어떤 채널이 연속형/이산형인지 파악 → 필요하면 이산형 채널은 원-핫 인코딩
3. 채널 간 상관관계 분석 → 필요하면 PCA 등으로 차원 축소
4. 단순 모델부터 시도 - Isolation Forest, One-Class SVM, Local Outlier Factor, GMM, PCA-based 등
5. 개선 방향 선택


- 데이터 이해: 데이터의 구조, 형태 분포 파악 → EDA
- 전처리 근거 마련: 결측치, 이상치를 어떻게 처리할지, 스케일링이 필요한지 등
- 가설 검증/수정: 데이터에 기반하여 새로운 가설을 세우고 검증 → 기존 가설이 틀렸다면 수정
- 모델 선정: 데이터의 특성에 맞는 모델 선택 → Isolation Forest, One-Class SVM, LOF, GMM, PCA-based 등
- 모델 개선: 모델의 성능을 높이기 위한 방법 탐색 → 하이퍼파라미터 튜닝, 앙상블, 추가 피처 엔지니어링 등
- 연속형과 이산형을 구분하여 각각에 맞는 전처리 적용 → 예: 연속형은 스케일링, 이산형은 원-핫 인코딩
- AUROC 0.7~0.8, AUPR 0.5~0.6 이상 목표 → 모델 개선 방향 설정에 참고


**로직**
- 다변량 시계열 데이터가 있음: timestep `t`, 채널 `x_xx` 10개, 라벨 `label` (val.csv, test_public.csv에만)
- sliding window로 데이터를 하나의 timestep이 아닌 window 단위로 변환
- window 단위로 anomaly score를 계산 
- window 단위 score를 timestep 단위로 환산 (예: 각 timestep이 포함된 모든 window의 score 평균)
```
슬라이딩 윈도우 방식으로 데이터를 자르면, 중간에 있는 특정 타임스텝(예: t=50)은 
여러 윈도우(예: 윈도우 48, 49, 50)에 중복해서 포함됩니다.
따라서 해당 타임스텝의 최종 점수는 자신을 포함했던 여러 윈도우들이
받은 점수들의 '평균(Average)'이나 '최댓값(Max)'을 계산해서 하나의 스칼라 값으로 정해줍니다.
```
- 즉, 모델은 window 단위로 학습/예측하지만 최종 제출은 timestep 단위로 해야 함
- timestep 단위는 스칼라 값이므로, 모델에서 window 단위로 나온 score를 timestep 단위로 환산
- 구한 timestap을 validation/test set의 라벨과 비교하여 AUROC, AUPR 계산 (정상: 0, 이상: 1)

**알아야할 점:**
- 모델마다 window 단위로 score를 내놓는 방식이 다를 수 있음 (예: Isolation Forest는 anomaly score, One-Class SVM은 decision function 등)
- Isolation Forest의 경우, `score_samples` 메서드로 window 단위 anomaly score를 얻을 수 있음 (클수록 이상)
- 왜냐하면 함수 `model.score_samples()`나 `model.decision_function()`이 window 단위로 anomaly score를 반환하기 때문
- 하지만 다른 모델은 윈도우별 이상 점수를 계산하기 위해서 추가적인 로직이 필요할 수 있음 (K-means의 경우, 각 윈도우가 가장 가까운 클러스터 중심에서 얼마나 멀리 떨어져 있는지 계산하는 식으로)
- AUROC, AUPR 계산을 위해서 이상치 score가 클수록 이상으로 간주되도록 해야 함 (일부 모델은 반대로 나올 수 있으니 주의)
- 즉, 정상보다 이상이 더 높은 score를 갖도록 모델의 출력이나 후처리를 조정해야 할 수 있음
- 그리고 앙상블 시, 각 모델의 window 단위 score를 동일한 방향으로 맞추고 0과 1 사이로 정규화한 후 평균을 내는 방식으로 진행할 수 있음 -> 그렇지 않으면 모델마다 score의 스케일과 방향이 달라서 앙상블이 어려울 수 있음

---

# 이상탐지 과제 데이터셋

## 디렉토리 구성

```
project/
├── starter.py              # 데이터 로드, sliding window helper
├── data/
│   ├── train.csv                       # 학습용 (정상만)
│   ├── val.csv                         # 검증용 (정상+이상, 라벨 포함)
│   ├── test_public.csv                 # 자체 평가용 (정상+이상, 라벨 포함)
└── └── test_hidden_no_labels.csv       # 최종 제출용 (라벨 없음)
```

## 데이터 설명

- **다변량 시계열**, 일정한 간격으로 샘플링됨
- 컬럼: `t` (timestep, 정수) + `x_xx` 형식의 채널 10개 + `label` (일부 파일만)
- 일부 채널은 **연속형**, 일부는 **이산형**입니다. 어느 것이 어느 종류인지는 EDA로 직접 파악하세요.
- 채널 간에 의존성/상관이 존재할 수 있습니다.
- 정상 데이터에는 어떤 주기적 패턴이 있을 수 있습니다.

## 파일별 라벨 정책

| 파일 | 라벨 | 용도 |
|---|---|---|
| `train.csv` | 없음 (모두 정상) | 모델 학습 |
| `val.csv` | 있음 (0=정상, 1=이상) | 하이퍼파라미터 튜닝 |
| `test_public.csv` | 있음 | 자체 성능 검증 |
| `test_hidden_no_labels.csv` | 없음 | 최종 제출 (강사 비공개 채점) |

## starter.py

데이터 로드와 sliding window 변환을 도와주는 helper들이 들어 있습니다.

helper로 제공되는 것:
- `load_split(name)`: 한 split 로드
- `make_windows(values, window_size, stride)`: sliding window
- `make_window_labels(labels, window_size, stride, policy)`: window 단위 라벨

직접 작성해야 하는 것:
- EDA 및 시각화
- 전처리 (스케일링, 이산/연속 분리, 추가 피처 등)
- 모델 학습 (예: Isolation Forest, One-Class SVM, LOF, GMM, PCA-based 등 전통 ML)
- 평가 (AUROC, AUPR — sklearn의 `roc_auc_score`, `average_precision_score`)
- window-단위 score를 timestep-단위로 환산하는 로직 (sliding window를 쓸 경우)

`starter.py`를 직접 실행하면 (`python starter.py`) 데이터 로드 → sliding window까지의 흐름을 한 번 보여줍니다.

## 평가

- **timestep 단위 AUROC, AUPR** 두 지표로 채점
- 라벨이 없는 `test_hidden_no_labels.csv`의 각 timestep에 대한 anomaly score를 제출
- score는 **클수록 이상**으로 간주됩니다

## 제출 형식

`submission.csv` 파일을 다음 형식으로 제출:

```
t,score
0,0.0123
1,0.0145
2,0.0119
...
```

- 컬럼은 정확히 `t`, `score` 두 개
- `t`는 `test_hidden_no_labels.csv`의 `t`와 동일한 timestep
- `score`는 실수값 (정규화 여부 무관, 상대적 순위만 사용됨)
