# 07 Cycle-Run 경계 재검증

## 목적

시간상 떨어진 시행에서 동일 `cycle` ID가 재등장하는 문제를 수정하고, 기존 `02-05` 결과가 `cycle_run` 경계에서도 유지되는지 확인한다.

## Window 수 변화

| target | window_size | windows_raw_cycle | windows_cycle_run | windows_removed | cycles | cycle_runs |
| --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 5 | 5624 | 5600 | 24 | 218 | 224 |
| System_Failure | 10 | 4536 | 4482 | 54 | 215 | 220 |
| System_Failure | 20 | 2411 | 2307 | 104 | 209 | 214 |
| ProtectiveStop | 5 | 5624 | 5600 | 24 | 218 | 224 |
| ProtectiveStop | 10 | 4536 | 4482 | 54 | 215 | 220 |
| ProtectiveStop | 20 | 2411 | 2307 | 104 | 209 | 214 |
| GripLost | 5 | 5624 | 5600 | 24 | 218 | 224 |
| GripLost | 10 | 4536 | 4482 | 54 | 215 | 220 |
| GripLost | 20 | 2411 | 2307 | 104 | 209 | 214 |

- `windows_removed`는 raw `cycle`로 합쳤을 때 서로 다른 시행 사이에 만들어졌던 경계-crossing window 수다.

## 02 구간 단위 이상탐지 비교

| target | window_size | model | macro_f1_raw_cycle | macro_f1_cycle_run | macro_f1_delta | positive_recall_raw_cycle | positive_recall_cycle_run | positive_recall_delta | positive_f1_raw_cycle | positive_f1_cycle_run | positive_f1_delta | pr_auc_raw_cycle | pr_auc_cycle_run | pr_auc_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 5 | rf_smote | 0.8511 | 0.8166 | -0.0345 | 0.6391 | 0.5556 | -0.0835 | 0.7423 | 0.6954 | -0.0468 | 0.8581 | 0.8613 | 0.0032 |
| System_Failure | 10 | rf_smote | 0.8593 | 0.8717 | 0.0124 | 0.6944 | 0.8447 | 0.1502 | 0.7883 | 0.8 | 0.0117 | 0.8936 | 0.8336 | -0.0599 |
| System_Failure | 20 | rf_smote | 0.6338 | 0.7425 | 0.1087 | 0.4541 | 0.7277 | 0.2737 | 0.5472 | 0.7 | 0.1528 | 0.677 | 0.6689 | -0.0082 |
| ProtectiveStop | 5 | rf_smote | 0.8374 | 0.8642 | 0.0267 | 0.5752 | 0.68 | 0.1048 | 0.7027 | 0.7628 | 0.0601 | 0.8445 | 0.9039 | 0.0594 |
| ProtectiveStop | 10 | rf_smote | 0.9323 | 0.8862 | -0.0461 | 0.8487 | 0.8611 | 0.0124 | 0.8866 | 0.8078 | -0.0788 | 0.9394 | 0.8292 | -0.1101 |
| ProtectiveStop | 20 | rf_smote | 0.7024 | 0.7858 | 0.0835 | 0.4928 | 0.7226 | 0.2298 | 0.5787 | 0.7066 | 0.1279 | 0.6498 | 0.694 | 0.0442 |
| GripLost | 5 | rf_smote | 0.8796 | 0.7646 | -0.115 | 0.6984 | 0.382 | -0.3164 | 0.7719 | 0.5528 | -0.2191 | 0.8148 | 0.67 | -0.1448 |
| GripLost | 10 | rf_smote | 0.7332 | 0.8412 | 0.108 | 0.3478 | 0.5806 | 0.2328 | 0.5128 | 0.699 | 0.1862 | 0.7189 | 0.8493 | 0.1303 |
| GripLost | 20 | rf_smote | 0.6899 | 0.7744 | 0.0844 | 0.2917 | 0.4412 | 0.1495 | 0.4516 | 0.5941 | 0.1424 | 0.6284 | 0.7609 | 0.1325 |

- 단일 group split 결과이므로 delta에는 경계 수정과 test 시행 구성 변화가 함께 반영된다.
- window별 순위가 바뀌면 기존 단일 best 설정을 고정 결론으로 사용하지 않고 block-held-out 또는 반복 검증 결과를 우선한다.

## 03 Pre-Failure 단일 Split Best 비교

| target | window_size_raw_cycle | prediction_horizon_raw_cycle | positive_recall_raw_cycle | positive_precision_raw_cycle | positive_f1_raw_cycle | pr_auc_raw_cycle | window_size_cycle_run | prediction_horizon_cycle_run | positive_recall_cycle_run | positive_precision_cycle_run | positive_f1_cycle_run | pr_auc_cycle_run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GripLost | 5 | 3 | 0.3333 | 0.6364 | 0.4375 | 0.5106 | 5 | 3 | 0.1667 | 1.0 | 0.2857 | 0.5099 |
| ProtectiveStop | 10 | 3 | 0.3333 | 1.0 | 0.5 | 0.5079 | 5 | 10 | 0.087 | 1.0 | 0.16 | 0.2124 |
| System_Failure | 5 | 10 | 0.0426 | 0.25 | 0.0727 | 0.28 | 5 | 10 | 0.12 | 0.4286 | 0.1875 | 0.3287 |

- best 조합 자체가 달라질 수 있으므로 이 표는 안정성 진단용이며 직접 성능 개선 주장에 사용하지 않는다.

## 04 동일 설정 30회 반복 비교

| target | window_size | prediction_horizon | model | runs_with_tp_rate_raw_cycle | runs_with_tp_rate_cycle_run | positive_recall_mean_raw_cycle | positive_recall_mean_cycle_run | positive_f1_mean_raw_cycle | positive_f1_mean_cycle_run | pr_auc_mean_raw_cycle | pr_auc_mean_cycle_run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 5 | 10 | rf_smote | 0.7333 | 0.8333 | 0.0749 | 0.0644 | 0.0963 | 0.092 | 0.2054 | 0.2062 |
| ProtectiveStop | 10 | 3 | rf_smote | 0.1 | 0.1333 | 0.05 | 0.0667 | 0.0656 | 0.0867 | 0.335 | 0.4151 |
| GripLost | 5 | 3 | rf_smote | 1.0 | 0.9667 | 0.2381 | 0.2248 | 0.3164 | 0.3058 | 0.3819 | 0.4127 |

- 동일 target/window/horizon을 30회 반복한 결과가 경계 수정 전후 결론 안정성을 판단하는 주 근거다.

## 05 Threshold 추천 비교

| target | threshold_raw_cycle | positive_recall_mean_raw_cycle | positive_f1_mean_raw_cycle | false_positive_rate_mean_raw_cycle | threshold_cycle_run | positive_recall_mean_cycle_run | positive_f1_mean_cycle_run | false_positive_rate_mean_cycle_run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GripLost | 0.3 | 0.4284 | 0.3803 | 0.0125 | 0.3 | 0.4396 | 0.4101 | 0.0118 |
| ProtectiveStop | 0.15 | 0.3481 | 0.2596 | 0.0168 | 0.2 | 0.3296 | 0.3231 | 0.0053 |
| System_Failure | 0.3 | 0.2669 | 0.2069 | 0.088 | 0.3 | 0.2211 | 0.1856 | 0.0846 |

## 결론

- raw `cycle` 경계 문제는 실제 window 구성과 일부 단일 split best 결과를 바꿨으므로 수정이 필요했다.
- 반복 검증의 positive F1 평균은 `GripLost` 0.3164 -> 0.3058, `ProtectiveStop` 0.0656 -> 0.0867, `System_Failure` 0.0963 -> 0.0920였다.
- 따라서 `GripLost`가 상대적으로 가장 안정적인 pre-failure 타깃이라는 결론은 유지되지만, 절대 성능은 여전히 낮아 약한 사전 신호로만 해석한다.
- threshold 후보는 `GripLost`와 `System_Failure`에서 0.30을 유지했고, `ProtectiveStop`은 0.15에서 0.20으로 바뀌어 고정값으로 보기 어렵다.
- 이후 결과 보고는 raw `cycle` 버전이 아니라 `cycle_run` 버전을 기준으로 한다.
