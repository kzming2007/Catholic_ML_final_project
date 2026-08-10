# 01 Baseline 재현

## 범위

- Feature set: 원본 센서 feature 19개 + current-speed power feature 6개 + absolute current sum.
- Models: class weighting을 적용한 Random Forest, SMOTE + Random Forest.
- Splits: row 단위 stratified random split, cycle 기준 group split.
- Targets: System_Failure, ProtectiveStop, GripLost.

## 결과

| target | split | model | test_rows | positive_count | positive_rate | accuracy | macro_f1 | positive_recall | positive_precision | positive_f1 | roc_auc | pr_auc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | random_stratified | rf_plain | 1471 | 104 | 0.0707 | 0.9449 | 0.6899 | 0.2692 | 0.8485 | 0.4088 | 0.9446 | 0.7039 | 1362 | 5 | 76 | 28 |
| System_Failure | random_stratified | rf_smote | 1471 | 104 | 0.0707 | 0.9511 | 0.8013 | 0.5865 | 0.6778 | 0.6289 | 0.9442 | 0.6601 | 1338 | 29 | 43 | 61 |
| System_Failure | cycle_group | rf_plain | 1436 | 94 | 0.0655 | 0.9464 | 0.673 | 0.2447 | 0.7931 | 0.374 | 0.9165 | 0.5741 | 1336 | 6 | 71 | 23 |
| System_Failure | cycle_group | rf_smote | 1436 | 94 | 0.0655 | 0.9519 | 0.7901 | 0.5638 | 0.6543 | 0.6057 | 0.9196 | 0.643 | 1314 | 28 | 41 | 53 |
| ProtectiveStop | random_stratified | rf_plain | 1471 | 56 | 0.0381 | 0.968 | 0.7086 | 0.3214 | 0.6667 | 0.4337 | 0.9767 | 0.5492 | 1406 | 9 | 38 | 18 |
| ProtectiveStop | random_stratified | rf_smote | 1471 | 56 | 0.0381 | 0.9667 | 0.7921 | 0.6607 | 0.5522 | 0.6016 | 0.9747 | 0.616 | 1385 | 30 | 19 | 37 |
| ProtectiveStop | cycle_group | rf_plain | 1436 | 45 | 0.0313 | 0.9756 | 0.716 | 0.3111 | 0.7778 | 0.4444 | 0.9802 | 0.6251 | 1387 | 4 | 31 | 14 |
| ProtectiveStop | cycle_group | rf_smote | 1436 | 45 | 0.0313 | 0.9805 | 0.8428 | 0.7111 | 0.6809 | 0.6957 | 0.9856 | 0.7409 | 1376 | 15 | 13 | 32 |
| GripLost | random_stratified | rf_plain | 1471 | 49 | 0.0333 | 0.9789 | 0.77 | 0.3878 | 0.95 | 0.5507 | 0.9558 | 0.7079 | 1421 | 1 | 30 | 19 |
| GripLost | random_stratified | rf_smote | 1471 | 49 | 0.0333 | 0.9748 | 0.7988 | 0.5918 | 0.6304 | 0.6105 | 0.9449 | 0.6943 | 1405 | 17 | 20 | 29 |
| GripLost | cycle_group | rf_plain | 1436 | 49 | 0.0341 | 0.9701 | 0.6511 | 0.2041 | 0.7143 | 0.3175 | 0.9176 | 0.4847 | 1383 | 4 | 39 | 10 |
| GripLost | cycle_group | rf_smote | 1436 | 49 | 0.0341 | 0.9714 | 0.7623 | 0.4898 | 0.6 | 0.5393 | 0.92 | 0.5604 | 1371 | 16 | 25 | 24 |

## 해석 메모

- random split보다 cycle group split에서 성능이 낮아지면 row 단위 random split이 낙관적이었을 가능성을 시사한다.
- `System_Failure`는 넓은 이상탐지 타깃으로 유용하지만, 고장 유형별 패턴을 보려면 ProtectiveStop과 GripLost를 별도로 분석해야 한다.
- 모든 타깃이 불균형하므로 accuracy보다 PR-AUC와 positive-class recall을 더 중요하게 본다.
