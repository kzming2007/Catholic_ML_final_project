# 14 센서 그룹 ablation 결과

## 고정 설계

- 사전 고정: `research/2026-08-23_sensor_group_ablation_preregistration.md`.
- 입력: cycle_run 경계를 넘지 않는 10-step×19개 원본 센서.
- 특징: 선택 센서마다 mean, std, min, max, range, delta, slope.
- 모델: SMOTE + Random Forest 300 trees, random_state 42, score > 0.50.
- 평가: 9개 acquisition block을 한 번씩 test로 사용.
- all_19 27개 fold는 기존 실험 12의 score와 prediction에 모두 일치했다.

## 핵심 결과

- `System_Failure` 전류 계열 제거: recall 0.9770->0.9310 (-0.0460), 오경보율 0.0435->0.1217 (+0.0783).
- `ProtectiveStop` 전류 계열 제거: recall 0.9516->0.9677 (+0.0161), 오경보율 0.0286->0.0786 (+0.0500).
- `GripLost` 전류 계열 제거: recall 0.9231->0.4103 (-0.5128), 오경보율 0.0368->0.0368 (+0.0000).
- `System_Failure` Tool current 제거: recall 0.9770->1.0000 (+0.0230), 오경보율 0.0435->0.0435 (+0.0000).
- `ProtectiveStop` Tool current 제거: recall 0.9516->0.9677 (+0.0161), 오경보율 0.0286->0.0286 (+0.0000).
- `GripLost` Tool current 제거: recall 0.9231->0.9231 (+0.0000), 오경보율 0.0368->0.0429 (+0.0061).
- Speed 단독은 event recall 0.9231-0.9839를 보였지만 정상 cycle 오경보율도 0.1214-0.3006였다.
- Temperature 단독 event recall은 0.0256-0.2529로 낮았다. 낮은 오경보와 함께 대부분의 event를 경보하지 않은 결과이므로 고장 탐지력이 높다고 해석하지 않는다.
- 종합하면 joint current는 System_Failure와 GripLost 구간 탐지에 중요한 정보를 제공하지만, 전류 계열이 모든 타깃에서 유일하거나 항상 최적인 것은 아니다. Tool current의 기여도 joint current와 분리해 해석해야 한다.

## All-sensor 기준선

| target | variant | feature_count | event_cycles | detected_event_cycles | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycles | false_alarm_cycles | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block | window_macro_f1 | window_positive_f1 | window_roc_auc | window_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | all_19 | 19 | 87 | 85 | 0.977 | 0.92 | 0.9937 | 0.95 | 115 | 5 | 0.0435 | 0.0187 | 0.0978 | 0.1667 | 0.8278 | 0.7341 | 0.9145 | 0.8137 |
| ProtectiveStop | all_19 | 19 | 62 | 59 | 0.9516 | 0.8671 | 0.9834 | 0.5 | 140 | 4 | 0.0286 | 0.0112 | 0.0712 | 0.1429 | 0.8755 | 0.7845 | 0.9594 | 0.8557 |
| GripLost | all_19 | 19 | 39 | 36 | 0.9231 | 0.7968 | 0.9735 | 0.6667 | 163 | 6 | 0.0368 | 0.017 | 0.078 | 0.15 | 0.7923 | 0.6195 | 0.9069 | 0.7267 |

## 센서 그룹 단독 사용

| target | variant | feature_count | event_cycles | detected_event_cycles | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycles | false_alarm_cycles | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block | window_macro_f1 | window_positive_f1 | window_roc_auc | window_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | current_family_only | 7 | 87 | 85 | 0.977 | 0.92 | 0.9937 | 0.8571 | 115 | 10 | 0.087 | 0.0479 | 0.1527 | 0.2222 | 0.8377 | 0.7499 | 0.912 | 0.8194 |
| System_Failure | joint_current_only | 6 | 87 | 86 | 0.9885 | 0.9377 | 0.998 | 0.95 | 115 | 14 | 0.1217 | 0.0739 | 0.194 | 0.3333 | 0.8347 | 0.747 | 0.9079 | 0.8144 |
| System_Failure | tool_current_only | 1 | 87 | 70 | 0.8046 | 0.7092 | 0.8743 | 0.5556 | 115 | 100 | 0.8696 | 0.7959 | 0.9193 | 0.9444 | 0.5074 | 0.248 | 0.5317 | 0.2535 |
| System_Failure | speed_only | 6 | 87 | 85 | 0.977 | 0.92 | 0.9937 | 0.8889 | 115 | 27 | 0.2348 | 0.1667 | 0.32 | 0.3889 | 0.7872 | 0.6779 | 0.86 | 0.7413 |
| System_Failure | temperature_only | 6 | 87 | 22 | 0.2529 | 0.1733 | 0.3533 | 0.0 | 115 | 9 | 0.0783 | 0.0417 | 0.1421 | 0.8333 | 0.501 | 0.1614 | 0.5503 | 0.28 |
| ProtectiveStop | current_family_only | 7 | 62 | 58 | 0.9355 | 0.8455 | 0.9746 | 0.6667 | 140 | 10 | 0.0714 | 0.0393 | 0.1265 | 0.1429 | 0.8686 | 0.7734 | 0.9634 | 0.8274 |
| ProtectiveStop | joint_current_only | 6 | 62 | 59 | 0.9516 | 0.8671 | 0.9834 | 0.8824 | 140 | 9 | 0.0643 | 0.0342 | 0.1177 | 0.1429 | 0.8659 | 0.7685 | 0.9615 | 0.8236 |
| ProtectiveStop | tool_current_only | 1 | 62 | 42 | 0.6774 | 0.5537 | 0.7805 | 0.3333 | 140 | 86 | 0.6143 | 0.5316 | 0.6908 | 0.7143 | 0.5706 | 0.2682 | 0.6413 | 0.2228 |
| ProtectiveStop | speed_only | 6 | 62 | 61 | 0.9839 | 0.9141 | 0.9971 | 0.9412 | 140 | 17 | 0.1214 | 0.0772 | 0.1859 | 0.2727 | 0.8137 | 0.6845 | 0.9296 | 0.755 |
| ProtectiveStop | temperature_only | 6 | 62 | 10 | 0.1613 | 0.09 | 0.2721 | 0.0 | 140 | 4 | 0.0286 | 0.0112 | 0.0712 | 0.2222 | 0.4813 | 0.0462 | 0.6394 | 0.2023 |
| GripLost | current_family_only | 7 | 39 | 37 | 0.9487 | 0.8311 | 0.9858 | 0.8 | 163 | 20 | 0.1227 | 0.0809 | 0.1819 | 0.381 | 0.7942 | 0.6282 | 0.9046 | 0.7088 |
| GripLost | joint_current_only | 6 | 39 | 37 | 0.9487 | 0.8311 | 0.9858 | 0.6667 | 163 | 18 | 0.1104 | 0.071 | 0.1678 | 0.381 | 0.8033 | 0.6461 | 0.9049 | 0.7127 |
| GripLost | tool_current_only | 1 | 39 | 22 | 0.5641 | 0.4098 | 0.707 | 0.25 | 163 | 127 | 0.7791 | 0.7094 | 0.836 | 0.9 | 0.48 | 0.0884 | 0.4798 | 0.1002 |
| GripLost | speed_only | 6 | 39 | 36 | 0.9231 | 0.7968 | 0.9735 | 0.6667 | 163 | 49 | 0.3006 | 0.2355 | 0.3749 | 0.5 | 0.7165 | 0.4936 | 0.7807 | 0.522 |
| GripLost | temperature_only | 6 | 39 | 1 | 0.0256 | 0.0045 | 0.1318 | 0.0 | 163 | 4 | 0.0245 | 0.0096 | 0.0614 | 0.2 | 0.4788 | 0.0173 | 0.5312 | 0.1203 |

## 센서 그룹 제거

| target | variant | feature_count | event_cycles | detected_event_cycles | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycles | false_alarm_cycles | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block | window_macro_f1 | window_positive_f1 | window_roc_auc | window_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | drop_current_family | 12 | 87 | 81 | 0.931 | 0.8576 | 0.968 | 0.7143 | 115 | 14 | 0.1217 | 0.0739 | 0.194 | 0.2857 | 0.7575 | 0.6174 | 0.8381 | 0.6754 |
| System_Failure | drop_speed | 13 | 87 | 84 | 0.9655 | 0.9035 | 0.9882 | 0.7143 | 115 | 4 | 0.0348 | 0.0136 | 0.086 | 0.1667 | 0.7769 | 0.6458 | 0.8905 | 0.7476 |
| System_Failure | drop_temperature | 13 | 87 | 86 | 0.9885 | 0.9377 | 0.998 | 0.95 | 115 | 10 | 0.087 | 0.0479 | 0.1527 | 0.2222 | 0.8521 | 0.7729 | 0.9196 | 0.8478 |
| System_Failure | drop_tool_current | 18 | 87 | 87 | 1.0 | 0.9577 | 1.0 | 1.0 | 115 | 5 | 0.0435 | 0.0187 | 0.0978 | 0.1667 | 0.8309 | 0.7387 | 0.9133 | 0.8126 |
| ProtectiveStop | drop_current_family | 12 | 62 | 60 | 0.9677 | 0.8898 | 0.9911 | 0.5 | 140 | 11 | 0.0786 | 0.0444 | 0.1352 | 0.1905 | 0.8263 | 0.7011 | 0.9161 | 0.7566 |
| ProtectiveStop | drop_speed | 13 | 62 | 57 | 0.9194 | 0.8247 | 0.9651 | 0.5 | 140 | 4 | 0.0286 | 0.0112 | 0.0712 | 0.25 | 0.8568 | 0.7507 | 0.9561 | 0.8145 |
| ProtectiveStop | drop_temperature | 13 | 62 | 59 | 0.9516 | 0.8671 | 0.9834 | 0.5 | 140 | 12 | 0.0857 | 0.0497 | 0.1438 | 0.3182 | 0.8821 | 0.7972 | 0.9673 | 0.8517 |
| ProtectiveStop | drop_tool_current | 18 | 62 | 60 | 0.9677 | 0.8898 | 0.9911 | 0.9412 | 140 | 4 | 0.0286 | 0.0112 | 0.0712 | 0.1429 | 0.8743 | 0.783 | 0.9587 | 0.8507 |
| GripLost | drop_current_family | 12 | 39 | 16 | 0.4103 | 0.2708 | 0.5658 | 0.0 | 163 | 6 | 0.0368 | 0.017 | 0.078 | 0.1053 | 0.6039 | 0.2587 | 0.7493 | 0.4 |
| GripLost | drop_speed | 13 | 39 | 32 | 0.8205 | 0.6733 | 0.9102 | 0.2 | 163 | 4 | 0.0245 | 0.0096 | 0.0614 | 0.2 | 0.7278 | 0.4976 | 0.8946 | 0.6541 |
| GripLost | drop_temperature | 13 | 39 | 37 | 0.9487 | 0.8311 | 0.9858 | 0.6667 | 163 | 12 | 0.0736 | 0.0426 | 0.1243 | 0.25 | 0.8261 | 0.685 | 0.9062 | 0.7551 |
| GripLost | drop_tool_current | 18 | 39 | 36 | 0.9231 | 0.7968 | 0.9735 | 0.6667 | 163 | 7 | 0.0429 | 0.021 | 0.086 | 0.15 | 0.795 | 0.6246 | 0.9058 | 0.739 |

## All-sensor 대비 변화량

| target | variant | event_cycle_recall_delta | normal_cycle_false_alarm_rate_delta | window_macro_f1_delta | window_pr_auc_delta |
| --- | --- | --- | --- | --- | --- |
| System_Failure | current_family_only | 0.0 | 0.0435 | 0.0099 | 0.0057 |
| System_Failure | joint_current_only | 0.0115 | 0.0783 | 0.0069 | 0.0007 |
| System_Failure | tool_current_only | -0.1724 | 0.8261 | -0.3204 | -0.5602 |
| System_Failure | speed_only | 0.0 | 0.1913 | -0.0406 | -0.0724 |
| System_Failure | temperature_only | -0.7241 | 0.0348 | -0.3268 | -0.5337 |
| System_Failure | drop_current_family | -0.046 | 0.0783 | -0.0703 | -0.1383 |
| System_Failure | drop_speed | -0.0115 | -0.0087 | -0.0509 | -0.0662 |
| System_Failure | drop_temperature | 0.0115 | 0.0435 | 0.0243 | 0.0341 |
| System_Failure | drop_tool_current | 0.023 | 0.0 | 0.0031 | -0.0011 |
| ProtectiveStop | current_family_only | -0.0161 | 0.0429 | -0.0069 | -0.0283 |
| ProtectiveStop | joint_current_only | 0.0 | 0.0357 | -0.0096 | -0.0321 |
| ProtectiveStop | tool_current_only | -0.2742 | 0.5857 | -0.3049 | -0.6328 |
| ProtectiveStop | speed_only | 0.0323 | 0.0929 | -0.0618 | -0.1007 |
| ProtectiveStop | temperature_only | -0.7903 | 0.0 | -0.3942 | -0.6533 |
| ProtectiveStop | drop_current_family | 0.0161 | 0.05 | -0.0492 | -0.0991 |
| ProtectiveStop | drop_speed | -0.0323 | 0.0 | -0.0187 | -0.0412 |
| ProtectiveStop | drop_temperature | 0.0 | 0.0571 | 0.0066 | -0.0039 |
| ProtectiveStop | drop_tool_current | 0.0161 | 0.0 | -0.0012 | -0.005 |
| GripLost | current_family_only | 0.0256 | 0.0859 | 0.0019 | -0.0178 |
| GripLost | joint_current_only | 0.0256 | 0.0736 | 0.011 | -0.014 |
| GripLost | tool_current_only | -0.359 | 0.7423 | -0.3123 | -0.6265 |
| GripLost | speed_only | 0.0 | 0.2638 | -0.0758 | -0.2046 |
| GripLost | temperature_only | -0.8974 | -0.0123 | -0.3135 | -0.6063 |
| GripLost | drop_current_family | -0.5128 | 0.0 | -0.1884 | -0.3266 |
| GripLost | drop_speed | -0.1026 | -0.0123 | -0.0645 | -0.0725 |
| GripLost | drop_temperature | 0.0256 | 0.0368 | 0.0338 | 0.0285 |
| GripLost | drop_tool_current | 0.0 | 0.0061 | 0.0027 | 0.0123 |

- Recall delta는 클수록 유리하고 정상 cycle 오경보율 delta는 작을수록 유리하다.
- Window 지표 delta와 cycle 지표 delta의 방향이 다르면 평가 단위에 따른 trade-off로 본다.

## All-sensor 대비 paired cycle 오류

| target | variant | error_type | eligible_cycles | all_19_error_count | variant_error_count | shared_error_count | new_variant_error_count | corrected_all_19_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | current_family_only | event_miss | 87 | 2 | 2 | 0 | 2 | 2 |
| System_Failure | current_family_only | false_alarm | 115 | 5 | 10 | 4 | 6 | 1 |
| System_Failure | joint_current_only | event_miss | 87 | 2 | 1 | 0 | 1 | 2 |
| System_Failure | joint_current_only | false_alarm | 115 | 5 | 14 | 4 | 10 | 1 |
| System_Failure | tool_current_only | event_miss | 87 | 2 | 17 | 0 | 17 | 2 |
| System_Failure | tool_current_only | false_alarm | 115 | 5 | 100 | 4 | 96 | 1 |
| System_Failure | speed_only | event_miss | 87 | 2 | 2 | 1 | 1 | 1 |
| System_Failure | speed_only | false_alarm | 115 | 5 | 27 | 5 | 22 | 0 |
| System_Failure | temperature_only | event_miss | 87 | 2 | 65 | 1 | 64 | 1 |
| System_Failure | temperature_only | false_alarm | 115 | 5 | 9 | 0 | 9 | 5 |
| System_Failure | drop_current_family | event_miss | 87 | 2 | 6 | 1 | 5 | 1 |
| System_Failure | drop_current_family | false_alarm | 115 | 5 | 14 | 5 | 9 | 0 |
| System_Failure | drop_speed | event_miss | 87 | 2 | 3 | 0 | 3 | 2 |
| System_Failure | drop_speed | false_alarm | 115 | 5 | 4 | 2 | 2 | 3 |
| System_Failure | drop_temperature | event_miss | 87 | 2 | 1 | 1 | 0 | 1 |
| System_Failure | drop_temperature | false_alarm | 115 | 5 | 10 | 5 | 5 | 0 |
| System_Failure | drop_tool_current | event_miss | 87 | 2 | 0 | 0 | 0 | 2 |
| System_Failure | drop_tool_current | false_alarm | 115 | 5 | 5 | 4 | 1 | 1 |
| ProtectiveStop | current_family_only | event_miss | 62 | 3 | 4 | 1 | 3 | 2 |
| ProtectiveStop | current_family_only | false_alarm | 140 | 4 | 10 | 3 | 7 | 1 |
| ProtectiveStop | joint_current_only | event_miss | 62 | 3 | 3 | 1 | 2 | 2 |
| ProtectiveStop | joint_current_only | false_alarm | 140 | 4 | 9 | 1 | 8 | 3 |
| ProtectiveStop | tool_current_only | event_miss | 62 | 3 | 20 | 1 | 19 | 2 |
| ProtectiveStop | tool_current_only | false_alarm | 140 | 4 | 86 | 2 | 84 | 2 |
| ProtectiveStop | speed_only | event_miss | 62 | 3 | 1 | 0 | 1 | 3 |
| ProtectiveStop | speed_only | false_alarm | 140 | 4 | 17 | 3 | 14 | 1 |
| ProtectiveStop | temperature_only | event_miss | 62 | 3 | 52 | 2 | 50 | 1 |
| ProtectiveStop | temperature_only | false_alarm | 140 | 4 | 4 | 0 | 4 | 4 |
| ProtectiveStop | drop_current_family | event_miss | 62 | 3 | 2 | 1 | 1 | 2 |
| ProtectiveStop | drop_current_family | false_alarm | 140 | 4 | 11 | 3 | 8 | 1 |
| ProtectiveStop | drop_speed | event_miss | 62 | 3 | 5 | 2 | 3 | 1 |
| ProtectiveStop | drop_speed | false_alarm | 140 | 4 | 4 | 1 | 3 | 3 |
| ProtectiveStop | drop_temperature | event_miss | 62 | 3 | 3 | 3 | 0 | 0 |
| ProtectiveStop | drop_temperature | false_alarm | 140 | 4 | 12 | 4 | 8 | 0 |
| ProtectiveStop | drop_tool_current | event_miss | 62 | 3 | 2 | 2 | 0 | 1 |
| ProtectiveStop | drop_tool_current | false_alarm | 140 | 4 | 4 | 4 | 0 | 0 |
| GripLost | current_family_only | event_miss | 39 | 3 | 2 | 1 | 1 | 2 |
| GripLost | current_family_only | false_alarm | 163 | 6 | 20 | 4 | 16 | 2 |
| GripLost | joint_current_only | event_miss | 39 | 3 | 2 | 2 | 0 | 1 |
| GripLost | joint_current_only | false_alarm | 163 | 6 | 18 | 3 | 15 | 3 |
| GripLost | tool_current_only | event_miss | 39 | 3 | 17 | 3 | 14 | 0 |
| GripLost | tool_current_only | false_alarm | 163 | 6 | 127 | 6 | 121 | 0 |
| GripLost | speed_only | event_miss | 39 | 3 | 3 | 2 | 1 | 1 |
| GripLost | speed_only | false_alarm | 163 | 6 | 49 | 4 | 45 | 2 |
| GripLost | temperature_only | event_miss | 39 | 3 | 38 | 3 | 35 | 0 |
| GripLost | temperature_only | false_alarm | 163 | 6 | 4 | 0 | 4 | 6 |
| GripLost | drop_current_family | event_miss | 39 | 3 | 23 | 3 | 20 | 0 |
| GripLost | drop_current_family | false_alarm | 163 | 6 | 6 | 0 | 6 | 6 |
| GripLost | drop_speed | event_miss | 39 | 3 | 7 | 3 | 4 | 0 |
| GripLost | drop_speed | false_alarm | 163 | 6 | 4 | 1 | 3 | 5 |
| GripLost | drop_temperature | event_miss | 39 | 3 | 2 | 2 | 0 | 1 |
| GripLost | drop_temperature | false_alarm | 163 | 6 | 12 | 6 | 6 | 0 |
| GripLost | drop_tool_current | event_miss | 39 | 3 | 3 | 3 | 0 | 0 |
| GripLost | drop_tool_current | false_alarm | 163 | 6 | 7 | 5 | 2 | 1 |

## 사전 고정 해석 규칙 적용

- `System_Failure` 단독 그룹: current_family_only가 speed_only와 temperature_only를 모두 Pareto 우세하지는 않아 전류 단독 우위를 일반화할 수 없다.
- `System_Failure` `drop_current_family`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `System_Failure` `drop_speed`: Recall과 오경보 방향이 엇갈리는 trade-off다.
- `System_Failure` `drop_temperature`: Recall과 오경보 방향이 엇갈리는 trade-off다.
- `System_Failure` `drop_tool_current`: drop_tool_current가 recall과 오경보 기준에서 기술적으로 우세하다.
- `ProtectiveStop` 단독 그룹: current_family_only가 speed_only와 temperature_only를 모두 Pareto 우세하지는 않아 전류 단독 우위를 일반화할 수 없다.
- `ProtectiveStop` `drop_current_family`: Recall과 오경보 방향이 엇갈리는 trade-off다.
- `ProtectiveStop` `drop_speed`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `ProtectiveStop` `drop_temperature`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `ProtectiveStop` `drop_tool_current`: drop_tool_current가 recall과 오경보 기준에서 기술적으로 우세하다.
- `GripLost` 단독 그룹: current_family_only가 speed_only와 temperature_only를 모두 Pareto 우세하지는 않아 전류 단독 우위를 일반화할 수 없다.
- `GripLost` `drop_current_family`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `GripLost` `drop_speed`: Recall과 오경보 방향이 엇갈리는 trade-off다.
- `GripLost` `drop_temperature`: Recall과 오경보 방향이 엇갈리는 trade-off다.
- `GripLost` `drop_tool_current`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- Pareto 우세는 recall과 오경보의 방향을 함께 본 기술적 비교이며 통계적 유의성이나 인과관계를 뜻하지 않는다.

## 실행 범위

- Random Forest run: 270개.
- Window prediction: 121,050개.
- 기록된 학습 시간 합: 635.5초.

## 해석 제한

- Feature 수와 SMOTE 공간이 variant마다 달라 제거 효과를 순수한 인과 기여로 해석할 수 없다.
- Tool current의 구간 탐지 성능은 고장 발생 전 유용성을 뜻하지 않는다.
- Temperature 성능에는 thermal/session drift가 포함될 수 있다.
- 4,035개 window는 겹치며 독립 평가 단위는 202개 cycle이다.
- 9개 block은 실제 공정조건 정답이 아닌 수집 구간 proxy다.
- 단일 공개 데이터와 Random Forest 고정 seed 1회의 내부 비교다.
