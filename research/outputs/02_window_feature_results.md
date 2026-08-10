# 02 Window Feature Baseline

## 범위

- Focus: 분류 기반 이상탐지.
- Target: `System_Failure`, `ProtectiveStop`, `GripLost`.
- Window: 시간상 연속한 `cycle_run` 경계를 넘지 않는 5, 10, 20 step sliding window.
- Label: window 내부에 positive target이 하나라도 있으면 positive.
- Feature: mean, std, min, max, range, first-last delta, simple slope.
- Split: 같은 `cycle_run`이 train/test에 동시에 들어가지 않는 `cycle_run_group` split.
- Models: class weighting을 적용한 Random Forest, SMOTE + Random Forest.

## Window 데이터 요약

| target | window_size | windows | cycle_runs | cycle_ids | positive_windows | positive_rate | mean_positive_points_in_positive_window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 5 | 5600 | 224 | 218 | 975 | 0.1741 | 2.2554 |
| System_Failure | 10 | 4482 | 220 | 215 | 1099 | 0.2452 | 2.9809 |
| System_Failure | 20 | 2307 | 214 | 209 | 1007 | 0.4365 | 3.8232 |
| ProtectiveStop | 5 | 5600 | 224 | 218 | 565 | 0.1009 | 2.1221 |
| ProtectiveStop | 10 | 4482 | 220 | 215 | 653 | 0.1457 | 2.7534 |
| ProtectiveStop | 20 | 2307 | 214 | 209 | 714 | 0.3095 | 3.1499 |
| GripLost | 5 | 5600 | 224 | 218 | 422 | 0.0754 | 2.4052 |
| GripLost | 10 | 4482 | 220 | 215 | 472 | 0.1053 | 3.1928 |
| GripLost | 20 | 2307 | 214 | 209 | 357 | 0.1547 | 4.5602 |

## 결과

| target | window_size | model | test_windows | positive_count | positive_rate | accuracy | macro_f1 | positive_recall | positive_precision | positive_f1 | roc_auc | pr_auc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 5 | rf_plain | 1229 | 261 | 0.2124 | 0.8763 | 0.7614 | 0.4291 | 0.9739 | 0.5957 | 0.9451 | 0.8707 | 965 | 3 | 149 | 112 |
| System_Failure | 5 | rf_smote | 1229 | 261 | 0.2124 | 0.8967 | 0.8166 | 0.5556 | 0.9295 | 0.6954 | 0.9424 | 0.8613 | 957 | 11 | 116 | 145 |
| System_Failure | 10 | rf_plain | 985 | 206 | 0.2091 | 0.9025 | 0.8376 | 0.6456 | 0.8526 | 0.7348 | 0.9474 | 0.8351 | 756 | 23 | 73 | 133 |
| System_Failure | 10 | rf_smote | 985 | 206 | 0.2091 | 0.9117 | 0.8717 | 0.8447 | 0.7598 | 0.8 | 0.9491 | 0.8336 | 724 | 55 | 32 | 174 |
| System_Failure | 20 | rf_plain | 503 | 202 | 0.4016 | 0.7296 | 0.7173 | 0.6485 | 0.6684 | 0.6583 | 0.8074 | 0.6716 | 236 | 65 | 71 | 131 |
| System_Failure | 20 | rf_smote | 503 | 202 | 0.4016 | 0.7495 | 0.7425 | 0.7277 | 0.6743 | 0.7 | 0.801 | 0.6689 | 230 | 71 | 55 | 147 |
| ProtectiveStop | 5 | rf_plain | 1229 | 175 | 0.1424 | 0.9186 | 0.7878 | 0.4686 | 0.9213 | 0.6212 | 0.9818 | 0.9036 | 1047 | 7 | 93 | 82 |
| ProtectiveStop | 5 | rf_smote | 1229 | 175 | 0.1424 | 0.9398 | 0.8642 | 0.68 | 0.8686 | 0.7628 | 0.9805 | 0.9039 | 1036 | 18 | 56 | 119 |
| ProtectiveStop | 10 | rf_plain | 985 | 144 | 0.1462 | 0.9391 | 0.8712 | 0.7292 | 0.8333 | 0.7778 | 0.9681 | 0.8546 | 820 | 21 | 39 | 105 |
| ProtectiveStop | 10 | rf_smote | 985 | 144 | 0.1462 | 0.9401 | 0.8862 | 0.8611 | 0.7607 | 0.8078 | 0.965 | 0.8292 | 802 | 39 | 20 | 124 |
| ProtectiveStop | 20 | rf_plain | 503 | 155 | 0.3082 | 0.8012 | 0.7557 | 0.6 | 0.7099 | 0.6503 | 0.8702 | 0.7054 | 310 | 38 | 62 | 93 |
| ProtectiveStop | 20 | rf_smote | 503 | 155 | 0.3082 | 0.8151 | 0.7858 | 0.7226 | 0.6914 | 0.7066 | 0.8617 | 0.694 | 298 | 50 | 43 | 112 |
| GripLost | 5 | rf_plain | 1229 | 89 | 0.0724 | 0.9536 | 0.7523 | 0.3596 | 1.0 | 0.5289 | 0.898 | 0.6621 | 1140 | 0 | 57 | 32 |
| GripLost | 5 | rf_smote | 1229 | 89 | 0.0724 | 0.9552 | 0.7646 | 0.382 | 1.0 | 0.5528 | 0.8882 | 0.67 | 1140 | 0 | 55 | 34 |
| GripLost | 10 | rf_plain | 985 | 62 | 0.0629 | 0.9492 | 0.649 | 0.1935 | 1.0 | 0.3243 | 0.982 | 0.8332 | 923 | 0 | 50 | 12 |
| GripLost | 10 | rf_smote | 985 | 62 | 0.0629 | 0.9685 | 0.8412 | 0.5806 | 0.878 | 0.699 | 0.9743 | 0.8493 | 918 | 5 | 26 | 36 |
| GripLost | 20 | rf_plain | 503 | 68 | 0.1352 | 0.9085 | 0.7193 | 0.3235 | 1.0 | 0.4889 | 0.8917 | 0.7547 | 435 | 0 | 46 | 22 |
| GripLost | 20 | rf_smote | 503 | 68 | 0.1352 | 0.9185 | 0.7744 | 0.4412 | 0.9091 | 0.5941 | 0.906 | 0.7609 | 432 | 3 | 38 | 30 |

## Row Baseline 대비 비교

| target | feature_level | window_size | model | macro_f1 | positive_recall | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| System_Failure | row_baseline |  | rf_smote | 0.7901 | 0.5638 | 0.643 |
| System_Failure | best_window_feature | 5 | rf_smote | 0.8166 | 0.5556 | 0.8613 |
| ProtectiveStop | row_baseline |  | rf_smote | 0.8428 | 0.7111 | 0.7409 |
| ProtectiveStop | best_window_feature | 5 | rf_smote | 0.8642 | 0.68 | 0.9039 |
| GripLost | row_baseline |  | rf_smote | 0.7623 | 0.4898 | 0.5604 |
| GripLost | best_window_feature | 10 | rf_smote | 0.8412 | 0.5806 | 0.8493 |

## 해석 메모

- 이 실험은 cycle 전체를 하나의 표본으로 쓰지 않고, cycle 내부의 짧은 구간에서 시계열 통계 feature를 만든다.
- `cycle_run_group` split을 사용하므로 같은 연속 시행에서 만들어진 window가 train/test에 동시에 섞이지 않는다.
- 원본 `cycle` ID가 시간상 떨어진 구간에 재등장하더라도 서로 다른 `cycle_run`으로 처리한다.
- window label은 구간 단위 이상탐지 정의이므로, 고장 발생 전 예측 문제로 해석하지 않는다.
- `System_Failure`는 전체 이상탐지 타깃이고, `ProtectiveStop`과 `GripLost`는 고장 유형별 패턴 차이를 보기 위한 보조 타깃이다.
