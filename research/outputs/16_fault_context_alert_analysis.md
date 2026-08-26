# Fault-context alert metric correction

## 목적

기존 결과의 `normal_cycle_false_alarm_rate`는 실제로 각 target이 없는 모든 cycle을 분모로 사용했다. 따라서 `ProtectiveStop` 평가에는 `GripLost`만 발생한 cycle이, `GripLost` 평가에는 `ProtectiveStop`만 발생한 cycle이 포함됐다. 이 분석은 재학습이나 threshold 변경 없이 기존 cycle 결과를 다음 세 범주로 분리한다.

- `target_negative_alert_rate`: 해당 target이 없는 모든 cycle에서의 경보율
- `true_normal_false_alarm_rate`: 어떤 고장도 없는 cycle에서의 오경보율
- `cross_fault_alert_rate`: 해당 target은 없지만 다른 고장만 있는 cycle에서의 경보율

`System_Failure`는 두 개별 고장의 합집합이므로 교차 고장 범주가 없다.

## 모델 비교

| candidate | target | event_cycle_recall | target_negative_alert_rate | true_normal_false_alarm_rate | cross_fault_alert_rate | other_fault_only_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 1d_cnn_19_raw | System_Failure | 0.977 | 0.1913 | 0.1913 |  | 0 |
| 1d_cnn_19_raw | ProtectiveStop | 0.9355 | 0.1929 | 0.2087 | 0.12 | 25 |
| 1d_cnn_19_raw | GripLost | 0.9231 | 0.2454 | 0.1652 | 0.4375 | 48 |
| lstm_autoencoder_19_raw_q95 | System_Failure | 0.046 | 0.1304 | 0.1304 |  | 0 |
| lstm_autoencoder_19_raw_q95 | ProtectiveStop | 0.0645 | 0.1071 | 0.1304 | 0.0 | 25 |
| lstm_autoencoder_19_raw_q95 | GripLost | 0.0 | 0.1227 | 0.1304 | 0.1042 | 48 |
| rf_19_raw_window | System_Failure | 0.977 | 0.0435 | 0.0435 |  | 0 |
| rf_19_raw_window | ProtectiveStop | 0.9516 | 0.0286 | 0.0348 | 0.0 | 25 |
| rf_19_raw_window | GripLost | 0.9231 | 0.0368 | 0.0174 | 0.0833 | 48 |
| logistic_regression | System_Failure | 0.9885 | 0.5304 | 0.5304 |  | 0 |
| logistic_regression | ProtectiveStop | 0.9677 | 0.2714 | 0.2696 | 0.28 | 25 |
| logistic_regression | GripLost | 1.0 | 0.4908 | 0.4522 | 0.5833 | 48 |
| rbf_svm | System_Failure | 0.954 | 0.0957 | 0.0957 |  | 0 |
| rbf_svm | ProtectiveStop | 0.9032 | 0.0214 | 0.0174 | 0.04 | 25 |
| rbf_svm | GripLost | 0.9487 | 0.1779 | 0.0957 | 0.375 | 48 |

## 센서 그룹 ablation

| candidate | target | event_cycle_recall | target_negative_alert_rate | true_normal_false_alarm_rate | cross_fault_alert_rate | other_fault_only_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| all_19 | System_Failure | 0.977 | 0.0435 | 0.0435 |  | 0 |
| all_19 | ProtectiveStop | 0.9516 | 0.0286 | 0.0348 | 0.0 | 25 |
| all_19 | GripLost | 0.9231 | 0.0368 | 0.0174 | 0.0833 | 48 |
| current_family_only | System_Failure | 0.977 | 0.087 | 0.087 |  | 0 |
| current_family_only | ProtectiveStop | 0.9355 | 0.0714 | 0.0696 | 0.08 | 25 |
| current_family_only | GripLost | 0.9487 | 0.1227 | 0.0696 | 0.25 | 48 |
| drop_current_family | System_Failure | 0.931 | 0.1217 | 0.1217 |  | 0 |
| drop_current_family | ProtectiveStop | 0.9677 | 0.0786 | 0.0609 | 0.16 | 25 |
| drop_current_family | GripLost | 0.4103 | 0.0368 | 0.0348 | 0.0417 | 48 |
| drop_speed | System_Failure | 0.9655 | 0.0348 | 0.0348 |  | 0 |
| drop_speed | ProtectiveStop | 0.9194 | 0.0286 | 0.0348 | 0.0 | 25 |
| drop_speed | GripLost | 0.8205 | 0.0245 | 0.0261 | 0.0208 | 48 |
| drop_temperature | System_Failure | 0.9885 | 0.087 | 0.087 |  | 0 |
| drop_temperature | ProtectiveStop | 0.9516 | 0.0857 | 0.0696 | 0.16 | 25 |
| drop_temperature | GripLost | 0.9487 | 0.0736 | 0.0348 | 0.1667 | 48 |
| drop_tool_current | System_Failure | 1.0 | 0.0435 | 0.0435 |  | 0 |
| drop_tool_current | ProtectiveStop | 0.9677 | 0.0286 | 0.0348 | 0.0 | 25 |
| drop_tool_current | GripLost | 0.9231 | 0.0429 | 0.0174 | 0.1042 | 48 |
| joint_current_only | System_Failure | 0.9885 | 0.1217 | 0.1217 |  | 0 |
| joint_current_only | ProtectiveStop | 0.9516 | 0.0643 | 0.0609 | 0.08 | 25 |
| joint_current_only | GripLost | 0.9487 | 0.1104 | 0.0522 | 0.25 | 48 |
| speed_only | System_Failure | 0.977 | 0.2348 | 0.2348 |  | 0 |
| speed_only | ProtectiveStop | 0.9839 | 0.1214 | 0.087 | 0.28 | 25 |
| speed_only | GripLost | 0.9231 | 0.3006 | 0.2348 | 0.4583 | 48 |
| temperature_only | System_Failure | 0.2529 | 0.0783 | 0.0783 |  | 0 |
| temperature_only | ProtectiveStop | 0.1613 | 0.0286 | 0.0174 | 0.08 | 25 |
| temperature_only | GripLost | 0.0256 | 0.0245 | 0.0174 | 0.0417 | 48 |
| tool_current_only | System_Failure | 0.8046 | 0.8696 | 0.8696 |  | 0 |
| tool_current_only | ProtectiveStop | 0.6774 | 0.6143 | 0.5913 | 0.72 | 25 |
| tool_current_only | GripLost | 0.5641 | 0.7791 | 0.7565 | 0.8333 | 48 |

## 해석 제한

- 교차 고장 경보는 곧바로 오류라고 단정할 수 없다. 두 고장 유형이 일부 센서 패턴을 공유할 수 있기 때문이다.
- 반대로 target별 분류 성능으로 보고할 때는 교차 고장 경보를 해당 target의 정답으로 계산할 수 없다.
- 세 지표는 동일한 cycle 예측을 서로 다른 운영 질문에 맞춰 재집계한 값이며 독립적인 새 실험이 아니다.
- 현재 결과는 동일 데이터셋의 block-held-out 내부 검증이며 외부 로봇·외부 수집 세션 일반화 증거가 아니다.
