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
