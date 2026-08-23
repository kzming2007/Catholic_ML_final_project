# 15 동일 시계열 특징 기반 기본 분류 모델 비교

## 고정 설계

- 사전 고정: `research/2026-08-23_classical_model_comparison_preregistration.md`.
- 입력: cycle_run 경계를 넘지 않는 10-step×19개 원본 센서의 133개 통계 특징.
- 모델: Logistic Regression L2 C=1, RBF SVM C=1·gamma=scale, 기존 Random Forest 300 trees.
- 전처리: 기본 모델은 학습 fold 내부 StandardScaler와 SMOTE, Random Forest는 기존 SMOTE 결과 재사용.
- 평가: 9개 acquisition block을 한 번씩 test로 사용하고 score > 0.50 적용.
- 새 기본 모델 학습: 54회, 기록된 실행 시간 합 779.1초.

## Primary cycle 결과와 secondary window 결과

| target | model | event_cycles | detected_event_cycles | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycles | false_alarm_cycles | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block | window_macro_f1 | window_positive_f1 | window_roc_auc | window_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | random_forest | 87 | 85 | 0.977 | 0.92 | 0.9937 | 0.95 | 115 | 5 | 0.0435 | 0.0187 | 0.0978 | 0.1667 | 0.8278 | 0.7341 | 0.9145 | 0.8137 |
| System_Failure | logistic_regression | 87 | 86 | 0.9885 | 0.9377 | 0.998 | 0.95 | 115 | 61 | 0.5304 | 0.4397 | 0.6192 | 0.8571 | 0.6994 | 0.5701 | 0.796 | 0.5989 |
| System_Failure | rbf_svm | 87 | 83 | 0.954 | 0.8877 | 0.982 | 0.4286 | 115 | 11 | 0.0957 | 0.0543 | 0.1632 | 0.3333 | 0.7964 | 0.6837 | 0.8876 | 0.7748 |
| ProtectiveStop | random_forest | 62 | 59 | 0.9516 | 0.8671 | 0.9834 | 0.5 | 140 | 4 | 0.0286 | 0.0112 | 0.0712 | 0.1429 | 0.8755 | 0.7845 | 0.9594 | 0.8557 |
| ProtectiveStop | logistic_regression | 62 | 60 | 0.9677 | 0.8898 | 0.9911 | 0.875 | 140 | 38 | 0.2714 | 0.2046 | 0.3505 | 0.4 | 0.7902 | 0.6516 | 0.8997 | 0.6684 |
| ProtectiveStop | rbf_svm | 62 | 56 | 0.9032 | 0.8045 | 0.9549 | 0.5 | 140 | 3 | 0.0214 | 0.0073 | 0.0611 | 0.1429 | 0.8352 | 0.7122 | 0.9463 | 0.793 |
| GripLost | random_forest | 39 | 36 | 0.9231 | 0.7968 | 0.9735 | 0.6667 | 163 | 6 | 0.0368 | 0.017 | 0.078 | 0.15 | 0.7923 | 0.6195 | 0.9069 | 0.7267 |
| GripLost | logistic_regression | 39 | 39 | 1.0 | 0.9103 | 1.0 | 1.0 | 163 | 80 | 0.4908 | 0.4152 | 0.5669 | 0.8571 | 0.7089 | 0.5084 | 0.8632 | 0.5283 |
| GripLost | rbf_svm | 39 | 37 | 0.9487 | 0.8311 | 0.9858 | 0.6667 | 163 | 29 | 0.1779 | 0.1268 | 0.2438 | 0.5 | 0.7891 | 0.6195 | 0.8997 | 0.686 |

## Random Forest 대비 해석

- `System_Failure` `logistic_regression`: event recall 0.9885 (+0.0115), 정상 cycle 오경보율 0.5304 (+0.4870); recall과 오경보 방향이 엇갈리는 trade-off였다.
- `System_Failure` `rbf_svm`: event recall 0.9540 (-0.0230), 정상 cycle 오경보율 0.0957 (+0.0522); Random Forest가 두 cycle 지표에서 기술적으로 우세했다.
- `ProtectiveStop` `logistic_regression`: event recall 0.9677 (+0.0161), 정상 cycle 오경보율 0.2714 (+0.2429); recall과 오경보 방향이 엇갈리는 trade-off였다.
- `ProtectiveStop` `rbf_svm`: event recall 0.9032 (-0.0484), 정상 cycle 오경보율 0.0214 (-0.0071); recall과 오경보 방향이 엇갈리는 trade-off였다.
- `GripLost` `logistic_regression`: event recall 1.0000 (+0.0769), 정상 cycle 오경보율 0.4908 (+0.4540); recall과 오경보 방향이 엇갈리는 trade-off였다.
- `GripLost` `rbf_svm`: event recall 0.9487 (+0.0256), 정상 cycle 오경보율 0.1779 (+0.1411); recall과 오경보 방향이 엇갈리는 trade-off였다.

## Paired cycle 오류

| target | model | error_type | eligible_cycles | random_forest_error_count | model_error_count | shared_error_count | new_model_error_count | corrected_rf_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | logistic_regression | event_miss | 87 | 2 | 1 | 0 | 1 | 2 |
| System_Failure | logistic_regression | false_alarm | 115 | 5 | 61 | 3 | 58 | 2 |
| System_Failure | rbf_svm | event_miss | 87 | 2 | 4 | 0 | 4 | 2 |
| System_Failure | rbf_svm | false_alarm | 115 | 5 | 11 | 3 | 8 | 2 |
| ProtectiveStop | logistic_regression | event_miss | 62 | 3 | 2 | 1 | 1 | 2 |
| ProtectiveStop | logistic_regression | false_alarm | 140 | 4 | 38 | 4 | 34 | 0 |
| ProtectiveStop | rbf_svm | event_miss | 62 | 3 | 6 | 3 | 3 | 0 |
| ProtectiveStop | rbf_svm | false_alarm | 140 | 4 | 3 | 2 | 1 | 2 |
| GripLost | logistic_regression | event_miss | 39 | 3 | 0 | 0 | 0 | 3 |
| GripLost | logistic_regression | false_alarm | 163 | 6 | 80 | 5 | 75 | 1 |
| GripLost | rbf_svm | event_miss | 39 | 3 | 2 | 2 | 0 | 1 |
| GripLost | rbf_svm | false_alarm | 163 | 6 | 29 | 4 | 25 | 2 |

- Event miss는 실제 positive window를 하나도 잡지 못한 event cycle이다.
- False alarm은 정상 cycle의 window 중 하나 이상에서 경보한 경우다.
- `new_model_error_count`는 Random Forest가 맞혔지만 비교 모델이 틀린 cycle이다.
- `corrected_rf_error_count`는 Random Forest 오류를 비교 모델이 바로잡은 cycle이다.

## 해석 제한

- 같은 입력과 split을 사용했지만 Logistic Regression과 SVM에는 모델에 필요한 표준화를 적용했다.
- Hyperparameter와 threshold를 탐색하지 않았으므로 각 모델 계열의 최고 성능 비교가 아니다.
- SVM probability와 Random Forest probability는 산출 방식이 다르다.
- Cycle 지표를 우선하며 window 지표만 좋아진 경우 운영상 우위로 해석하지 않는다.
- 구간 탐지 결과이며 조기 고장 예측 또는 외부 일반화의 근거가 아니다.
- 이 결과를 본 뒤 모델, C, kernel, gamma 또는 threshold를 추가로 조정하지 않는다.
