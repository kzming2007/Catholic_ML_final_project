# 09 Event-Level Block-Held-Out Validation

## 목적

딥러닝 비교 전에 공통 10-step과 동일한 후보 block held-out 분할을 고정하고, window 성능 외에 고장 event cycle 탐지율과 정상 cycle 오경보를 평가한다.

## 고정 설계

- Window: 10 step.
- 평가 block: [1, 2, 3, 4, 5, 7, 8, 9, 10].
- 모델: `SMOTE + Random Forest`, positive decision rule `score > 0.50`.
- Feature set: 전체 sensor를 주 설정으로 두고 temperature 제거를 ablation으로 병기한다.
- Event detection: 실제 positive가 포함된 window 중 하나 이상을 positive로 예측한 event cycle의 비율.
- Normal-cycle false alarm: 실제 positive window가 없는 정상 cycle에서 하나 이상의 positive 예측이 발생한 비율.
- 제한: 25-cycle block은 실제 공정조건 정답표가 아니라 cycle 번호 기반 proxy다.
- 제한: 이 평가는 이상이 포함된 window 탐지이며 고장 발생 전 예측이 아니다.

## 요약

| target | feature_set | valid_test_blocks | event_cycle_count_total | event_cycle_recall_micro | event_cycle_recall_min | normal_cycle_count_total | normal_cycle_false_alarm_rate_micro | normal_cycle_false_alarm_rate_max | false_positive_windows_per_normal_cycle_micro | window_positive_f1_mean | window_pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | all_sensors | 9 | 87 | 1.0 | 1.0 | 115 | 0.0522 | 0.1667 | 0.113 | 0.7451 | 0.8243 |
| System_Failure | no_temperature | 9 | 87 | 1.0 | 1.0 | 115 | 0.0783 | 0.1765 | 0.1739 | 0.7839 | 0.8718 |
| ProtectiveStop | all_sensors | 9 | 62 | 0.9516 | 0.5 | 140 | 0.0429 | 0.1429 | 0.1 | 0.762 | 0.795 |
| ProtectiveStop | no_temperature | 9 | 62 | 0.9516 | 0.5 | 140 | 0.0929 | 0.2727 | 0.1643 | 0.7543 | 0.8068 |
| GripLost | all_sensors | 9 | 39 | 0.9487 | 0.6667 | 163 | 0.0429 | 0.2 | 0.0552 | 0.6741 | 0.821 |
| GripLost | no_temperature | 9 | 39 | 0.9487 | 0.6667 | 163 | 0.0798 | 0.2 | 0.227 | 0.7128 | 0.818 |

## 해석 기준

- `event_cycle_recall_micro`는 전체 event cycle을 합쳐 계산한 탐지율이다.
- `event_cycle_recall_min`은 가장 어려운 held-out block의 탐지율이다.
- `normal_cycle_false_alarm_rate_micro`는 정상 cycle 가운데 경보가 한 번 이상 발생한 cycle 비율이다.
- 겹치는 window 때문에 window false positive가 cycle 오경보로 누적될 수 있으므로 두 지표를 함께 본다.
- 이 고정 결과를 Random Forest 기준선으로 사용하고, 후속 sequence model도 같은 10-step과 block을 사용한다.
