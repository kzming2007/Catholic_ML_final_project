# 2026-08-23 기본 분류 모델 비교 사전 고정 기록

## 문서 목적

동일한 시계열 구간 특징과 block-held-out 평가에서 Logistic Regression, RBF SVM, Random Forest를 비교해, 현재 성능이 시계열 특징 정제에서 오는지 Random Forest의 모델 구조에서 오는지 구분한다. 전체 결과 실행 전에 입력, 전처리, 모델, 분할, 지표와 해석 규칙을 고정한다.

이 문서는 정식 외부 사전등록이 아니라 Git 커밋으로 시점을 보존하는 내부 사전 고정 기록이다.

## 연구 질문

> 동일한 10-step×19-sensor 시계열 구간 특징과 9개 block-held-out 평가에서 선형 Logistic Regression, 비선형 RBF SVM, Random Forest의 이상 event 탐지율과 정상 cycle 오경보율은 어떻게 다른가?

이 실험의 목적은 최고 성능 모델이나 hyperparameter를 사후 탐색하는 것이 아니다. 다음 두 효과를 구분하는 것이 목적이다.

1. 10-step 시계열을 통계 특징으로 정제한 것 자체가 기본 분류 모델에서도 유용한가?
2. Random Forest의 비선형 분기와 특징 상호작용이 추가적인 이점을 제공하는가?

## 데이터와 공통 입력

- 데이터: UCI UR3 CobotOps 공개 데이터셋.
- 입력: `cycle_run` 경계를 넘지 않는 10-step window.
- 원본 센서: `Current_J0`-`Current_J5`, `Temperature_J0`-`Temperature_J5`, `Speed_J0`-`Speed_J5`, `Tool_current`의 19개.
- 통계 특징: 각 센서의 mean, std, min, max, range, first-last delta, slope를 계산한 133개 특징.
- 대상 block: `1, 2, 3, 4, 5, 7, 8, 9, 10`.
- 평가 cycle: 202개.
  - `System_Failure`: event 87개, normal 115개.
  - `ProtectiveStop`: event 62개, normal 140개.
  - `GripLost`: event 39개, normal 163개.
- Window: 4,035개. Window가 겹치므로 독립 표본 수로 해석하지 않는다.
- 타깃: `System_Failure`, `ProtectiveStop`, `GripLost`를 각각 평가한다.

## 고정 모델

| 모델 | 전처리와 고정 설정 | 역할 |
| --- | --- | --- |
| Logistic Regression | `StandardScaler → SMOTE → LogisticRegression`, L2, `C=1`, `solver=saga`, `max_iter=5000`, `random_state=42` | 선형 결정 경계 기준선 |
| RBF SVM | `StandardScaler → SMOTE → SVC`, RBF, `C=1`, `gamma=scale`, `probability=True`, `random_state=42` | 거리 기반 비선형 결정 경계 기준선 |
| Random Forest | `SMOTE → RandomForestClassifier`, 300 trees, `random_state=42` | 기존 `12`의 고정 주 기준선 |

- Logistic Regression과 SVM은 수업 프로젝트에서 사용한 모델 계열과 기본 설정 범위를 계승한다.
- `C=1`은 두 모델의 표준 기본값이며 현재 데이터에서 선택한 최적값이 아니다.
- 거리와 계수에 의존하는 Logistic Regression과 SVM에는 표준화가 필요하므로 학습 fold 내부에서 `StandardScaler`를 적용한다.
- SMOTE도 학습 fold 내부에서만 적용하며 `k_neighbors=min(5, minority_count-1)`, `random_state=42`로 고정한다.
- SMOTE로 학습 표본을 균형화하므로 별도 `class_weight`는 적용하지 않는다.
- 모든 모델의 decision rule은 positive class score `> 0.50`이다.
- Hyperparameter tuning, threshold tuning, feature selection과 모델 추가는 수행하지 않는다.
- Random Forest는 새로 학습하지 않고 `12_matched_rf_window_predictions.csv`의 고정 결과를 재사용한다.

## 분할과 정합성 조건

- Outer 평가는 9개 block을 한 번씩 test로 사용하고 나머지 8개 block으로 학습한다.
- Test block은 scaler, SMOTE, 모델 학습에 사용하지 않는다.
- Logistic Regression과 SVM의 test window 메타데이터와 label은 같은 target·block의 Random Forest 기준선과 정확히 일치해야 한다.
- 각 모델은 target별로 9개 test block과 4,035개 window, 202개 cycle을 모두 포함해야 한다.
- 수렴 실패, 단일 class fold, 비유한 score 또는 정합성 불일치가 있으면 결과 해석 전에 중단한다.

## 평가 지표

### Primary cycle 지표

- Event cycle recall: 실제 positive window가 있는 cycle에서 positive window를 하나 이상 탐지한 비율.
- Normal cycle false-alarm rate: 실제 positive window가 없는 cycle에서 하나 이상의 window가 positive로 예측된 비율.
- 두 비율의 Wilson 95% confidence interval.
- Held-out block별 event recall 최솟값과 정상 cycle false-alarm rate 최댓값.

### Secondary window 지표

- Accuracy.
- Macro F1.
- Positive precision, recall, F1.
- ROC-AUC.
- PR-AUC.
- Confusion matrix.

### Paired cycle 오류 비교

Logistic Regression과 SVM을 Random Forest와 같은 cycle에서 비교한다.

- 양쪽이 함께 놓친 event.
- 비교 모델에서 새로 놓친 event와 새로 복구한 event.
- 양쪽이 함께 낸 정상 cycle 오경보.
- 비교 모델에서 새로 발생한 오경보와 해소된 오경보.

## 해석 규칙

- Logistic Regression이 높은 event recall과 낮은 정상 cycle 오경보를 함께 보이면, 현재 통계 특징에 선형적으로 구분 가능한 정보가 남는다는 근거로 해석한다.
- RBF SVM이 Logistic Regression보다 두 primary 지표에서 유리하면, 비선형 결정 경계의 추가 기여 가능성을 제시한다.
- Random Forest가 비교 모델보다 두 primary 지표에서 유리하면, 현재 표본과 특징에서 트리 기반 비선형 분기와 상호작용이 더 적합했다는 근거로 해석한다.
- Recall과 오경보율 방향이 엇갈리면 trade-off로 보고 단일 순위를 주장하지 않는다.
- 사전에 equivalence margin을 정하지 않았으므로 수치가 비슷해도 통계적 동등성을 주장하지 않는다.
- ROC-AUC나 window F1이 좋아도 정상 cycle 오경보율이 악화되면 운영상 우위로 해석하지 않는다.
- 결과가 좋은 비교 모델이 나오더라도 동일 데이터의 사후 선택이므로 주 기준선을 자동으로 교체하지 않는다.

## 고정 출력

- `research/15_classical_model_comparison.py`
- `research/15_classical_model_comparison_results.py`
- `research/outputs/15_classical_model_predictions.csv`
- `research/outputs/15_classical_model_runs.csv`
- `research/outputs/15_classical_model_window_metrics.csv`
- `research/outputs/15_classical_model_cycle_results.csv`
- `research/outputs/15_classical_model_block_metrics.csv`
- `research/outputs/15_classical_model_summary.csv`
- `research/outputs/15_classical_model_paired_errors.csv`
- `research/outputs/15_classical_model_comparison.md`

## 해석 제한

- 모델에 필요한 표준화 여부가 다르므로 전처리가 완전히 동일한 비교는 아니다. 다만 입력 window, 133개 특징, SMOTE, 분할과 평가 단위는 같다.
- SVM probability는 각 학습 fold 내부의 추가 확률 보정을 포함하므로 Random Forest score와 같은 확률 추정 방식은 아니다.
- Random Forest와 기본 모델 모두 seed 42의 고정 1회다. 모델 seed 변동을 평가하지 않는다.
- 9개 block은 실제 공정조건 정답이 아니라 수집 구간 proxy다.
- 구간 탐지 window에는 이미 발생한 이상 상태가 포함될 수 있으므로 조기 고장 예측 결과가 아니다.
- 동일 공개 데이터에 대한 내부 분석이며 외부 검증이 아니다.
- 이 결과를 보고 `C`, kernel, `gamma`, threshold, window 크기 또는 특징 조합을 바꾸지 않는다.
