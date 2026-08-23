# 11 시계열 모델 오류 분석

## 분석 질문

- 딥러닝 오경보와 미탐이 seed가 바뀌어도 같은 cycle에서 반복되는가?
- 오류가 특정 held-out block에 집중되는가?
- Random Forest와 딥러닝이 같은 cycle에서 오류를 내는가?
- 반복 오경보 window의 센서 요약값은 true-negative window와 기술적으로 어떤 차이를 보이는가?

## 반복 오류 기준

- Cycle consensus: 1D CNN/LSTM의 3개 seed 중 2개 이상이 같은 cycle 안에서 한 번 이상 경보하면 반복 cycle 경보로 정의한다.
- Window consensus: 3개 seed 중 2개 이상이 동일 window를 positive로 판단하면 반복 window 경보로 정의한다.
- Random Forest: `09`의 고정 all-sensors prediction 1회를 사용한다.
- Event miss: 실제 positive window가 있는 cycle에서 consensus 기준으로 positive window를 하나도 탐지하지 못한 경우다.

## Consensus 결과

| target | model | seed_count | consensus_required | normal_cycle_count | consensus_false_alarm_cycle_count | consensus_false_alarm_rate | event_cycle_count | consensus_detected_event_count | consensus_event_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GripLost | rf_window_features | 1 | 1 | 163 | 7 | 0.0429 | 39 | 37 | 0.9487 |
| GripLost | 1d_cnn | 3 | 2 | 163 | 50 | 0.3067 | 39 | 37 | 0.9487 |
| GripLost | lstm | 3 | 2 | 163 | 58 | 0.3558 | 39 | 38 | 0.9744 |
| ProtectiveStop | rf_window_features | 1 | 1 | 140 | 6 | 0.0429 | 62 | 59 | 0.9516 |
| ProtectiveStop | 1d_cnn | 3 | 2 | 140 | 26 | 0.1857 | 62 | 58 | 0.9355 |
| ProtectiveStop | lstm | 3 | 2 | 140 | 30 | 0.2143 | 62 | 56 | 0.9032 |
| System_Failure | rf_window_features | 1 | 1 | 115 | 6 | 0.0522 | 87 | 87 | 1.0 |
| System_Failure | 1d_cnn | 3 | 2 | 115 | 16 | 0.1391 | 87 | 84 | 0.9655 |
| System_Failure | lstm | 3 | 2 | 115 | 20 | 0.1739 | 87 | 86 | 0.9885 |

## 주요 결과

- `System_Failure` consensus 오경보율: Random Forest 0.0522, 1D CNN 0.1391, LSTM 0.1739.
- `ProtectiveStop` consensus 오경보율: Random Forest 0.0429, 1D CNN 0.1857, LSTM 0.2143.
- `GripLost` consensus 오경보율: Random Forest 0.0429, 1D CNN 0.3067, LSTM 0.3558.
- `System_Failure`에서 3개 seed 모두 경보한 정상 cycle: 1D CNN 10개, LSTM 9개.
- `ProtectiveStop`에서 3개 seed 모두 경보한 정상 cycle: 1D CNN 15개, LSTM 20개.
- `GripLost`에서 3개 seed 모두 경보한 정상 cycle: 1D CNN 32개, LSTM 35개.
- `System_Failure`에서 Random Forest는 정상 처리했지만 deep learning만 반복 오경보한 cycle: 1D CNN 15개, LSTM 19개.
- `ProtectiveStop`에서 Random Forest는 정상 처리했지만 deep learning만 반복 오경보한 cycle: 1D CNN 23개, LSTM 26개.
- `GripLost`에서 Random Forest는 정상 처리했지만 deep learning만 반복 오경보한 cycle: 1D CNN 47개, LSTM 54개.
- 가장 높은 block 오경보율은 `GripLost` `lstm` block 8의 0.8000였다.
- 2/3 seed consensus를 적용해도 deep learning의 오경보 격차가 유지되므로, 추가 모델 확대보다 block 변화와 정상 저변동 구간에 대한 오류 원인 분석이 우선이다.

## 오경보 집중 block 상위 12개

| target | model | test_block | normal_cycle_count | consensus_false_alarm_cycle_count | consensus_false_alarm_rate | event_cycle_count | consensus_detected_event_count | consensus_event_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GripLost | lstm | 8 | 10 | 8 | 0.8 | 2 | 2 | 1.0 |
| GripLost | 1d_cnn | 10 | 21 | 16 | 0.7619 | 6 | 6 | 1.0 |
| System_Failure | lstm | 10 | 7 | 5 | 0.7143 | 20 | 20 | 1.0 |
| System_Failure | 1d_cnn | 9 | 6 | 4 | 0.6667 | 20 | 18 | 0.9 |
| System_Failure | 1d_cnn | 10 | 7 | 4 | 0.5714 | 20 | 19 | 0.95 |
| GripLost | 1d_cnn | 2 | 20 | 11 | 0.55 | 3 | 3 | 1.0 |
| GripLost | lstm | 10 | 21 | 11 | 0.5238 | 6 | 5 | 0.8333 |
| ProtectiveStop | 1d_cnn | 10 | 8 | 4 | 0.5 | 19 | 19 | 1.0 |
| ProtectiveStop | lstm | 10 | 8 | 4 | 0.5 | 19 | 19 | 1.0 |
| System_Failure | lstm | 9 | 6 | 3 | 0.5 | 20 | 20 | 1.0 |
| ProtectiveStop | 1d_cnn | 5 | 20 | 9 | 0.45 | 5 | 5 | 1.0 |
| GripLost | lstm | 2 | 20 | 9 | 0.45 | 3 | 3 | 1.0 |

## Seed 일관성

| target | model | normal_cycle_count | event_cycle_count | normal_cycles_alerted_by_0_seeds | event_cycles_detected_by_0_seeds | normal_cycles_alerted_by_1_seeds | event_cycles_detected_by_1_seeds | normal_cycles_alerted_by_2_seeds | event_cycles_detected_by_2_seeds | normal_cycles_alerted_by_3_seeds | event_cycles_detected_by_3_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 1d_cnn | 115 | 87 | 86 | 0 | 13 | 3 | 6 | 1 | 10 | 83 |
| System_Failure | lstm | 115 | 87 | 73 | 1 | 22 | 0 | 11 | 7 | 9 | 79 |
| ProtectiveStop | 1d_cnn | 140 | 62 | 96 | 2 | 18 | 2 | 11 | 2 | 15 | 56 |
| ProtectiveStop | lstm | 140 | 62 | 101 | 5 | 9 | 1 | 10 | 0 | 20 | 56 |
| GripLost | 1d_cnn | 163 | 39 | 86 | 0 | 27 | 2 | 18 | 2 | 32 | 35 |
| GripLost | lstm | 163 | 39 | 72 | 0 | 33 | 1 | 23 | 4 | 35 | 34 |

## Random Forest 대비 오류 겹침

| target | error_type | deep_model | eligible_cycle_count | rf_error_count | deep_consensus_error_count | shared_error_count | deep_only_error_count | rf_only_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GripLost | false_alarm | 1d_cnn | 163 | 7 | 50 | 3 | 47 | 4 |
| GripLost | false_alarm | lstm | 163 | 7 | 58 | 4 | 54 | 3 |
| ProtectiveStop | false_alarm | 1d_cnn | 140 | 6 | 26 | 3 | 23 | 3 |
| ProtectiveStop | false_alarm | lstm | 140 | 6 | 30 | 4 | 26 | 2 |
| System_Failure | false_alarm | 1d_cnn | 115 | 6 | 16 | 1 | 15 | 5 |
| System_Failure | false_alarm | lstm | 115 | 6 | 20 | 1 | 19 | 5 |
| GripLost | event_miss | 1d_cnn | 39 | 2 | 2 | 0 | 2 | 2 |
| GripLost | event_miss | lstm | 39 | 2 | 1 | 1 | 0 | 1 |
| ProtectiveStop | event_miss | 1d_cnn | 62 | 3 | 4 | 1 | 3 | 2 |
| ProtectiveStop | event_miss | lstm | 62 | 3 | 6 | 2 | 4 | 1 |
| System_Failure | event_miss | 1d_cnn | 87 | 0 | 3 | 0 | 3 | 0 |
| System_Failure | event_miss | lstm | 87 | 0 | 1 | 0 | 1 | 0 |

`deep_only_error_count`가 크면 Random Forest가 틀리지 않은 cycle에서 딥러닝만 반복적으로 틀렸다는 뜻이다.

## 정상 cycle 내 반복 오경보 window의 센서 차이 상위 3개

| target | model | feature | window_statistic | false_alarm_window_count | true_negative_window_count | false_alarm_median | true_negative_median | robust_shift_iqr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | rf_window_features | Speed_J3 | range | 13 | 1714 | 0.6693 | 1.5591 | -1.7702 |
| System_Failure | rf_window_features | Speed_J5 | range | 13 | 1714 | 0.5183 | 1.3001 | -1.7555 |
| System_Failure | rf_window_features | Current_J5 | mean | 13 | 1714 | 0.0462 | -0.0064 | 1.3367 |
| System_Failure | 1d_cnn | Current_J3 | mean | 43 | 1684 | -0.7333 | -0.5674 | -0.9294 |
| System_Failure | 1d_cnn | Temperature_J0 | mean | 43 | 1684 | 37.0 | 33.8656 | 0.5776 |
| System_Failure | 1d_cnn | Temperature_J1 | mean | 43 | 1684 | 39.9812 | 36.4094 | 0.5474 |
| System_Failure | lstm | Tool_current | range | 41 | 1686 | 0.3984 | 0.1121 | 0.9768 |
| System_Failure | lstm | Power_J2 | range | 41 | 1686 | 0.1951 | 3.9787 | -0.924 |
| System_Failure | lstm | Tool_current | mean | 41 | 1686 | 0.1309 | 0.1004 | 0.8605 |
| ProtectiveStop | rf_window_features | Current_J5 | mean | 14 | 2128 | 0.0494 | -0.0063 | 1.3829 |
| ProtectiveStop | rf_window_features | Speed_J5 | range | 14 | 2128 | 0.7215 | 1.2736 | -1.1362 |
| ProtectiveStop | rf_window_features | Speed_J3 | range | 14 | 2128 | 0.9443 | 1.5379 | -1.0531 |
| ProtectiveStop | 1d_cnn | Current_J2 | mean | 46 | 2096 | -1.359 | -1.1812 | -1.0319 |
| ProtectiveStop | 1d_cnn | Current_J3 | mean | 46 | 2096 | -0.6974 | -0.5644 | -0.9127 |
| ProtectiveStop | 1d_cnn | Speed_J4 | range | 46 | 2096 | 0.0423 | 0.1474 | -0.9122 |
| ProtectiveStop | lstm | Current_J3 | mean | 50 | 2092 | -0.7285 | -0.565 | -1.1286 |
| ProtectiveStop | lstm | Current_J2 | mean | 50 | 2092 | -1.3398 | -1.1806 | -0.9215 |
| ProtectiveStop | lstm | Speed_J4 | range | 50 | 2092 | 0.0443 | 0.1474 | -0.894 |
| GripLost | rf_window_features | Temperature_J1 | range | 9 | 3199 | 0.0 | 0.0625 | -1.0 |
| GripLost | rf_window_features | Abs_Current_Sum | range | 9 | 3199 | 1.8353 | 5.3511 | -0.9377 |
| GripLost | rf_window_features | Abs_Current_Sum | mean | 9 | 3199 | 4.2068 | 4.953 | -0.7398 |
| GripLost | 1d_cnn | Abs_Current_Sum | range | 228 | 2980 | 2.5863 | 5.4213 | -0.8183 |
| GripLost | 1d_cnn | Abs_Current_Sum | mean | 228 | 2980 | 4.2604 | 5.0048 | -0.7585 |
| GripLost | 1d_cnn | Current_J2 | mean | 228 | 2980 | -1.0616 | -1.2124 | 0.6957 |
| GripLost | lstm | Abs_Current_Sum | mean | 204 | 3004 | 4.1603 | 4.9915 | -0.8535 |
| GripLost | lstm | Abs_Current_Sum | range | 204 | 3004 | 2.521 | 5.4193 | -0.838 |
| GripLost | lstm | Current_J2 | mean | 204 | 3004 | -1.0518 | -1.2122 | 0.7402 |

## 해석 제한

- sensor shift는 전체 window가 음성인 정상 cycle만 사용한다. 같은 cycle에서 겹치는 window가 다수 생성되므로 독립표본 유의성 검정이 아니라 오류 원인을 찾기 위한 기술통계다.
- `robust_shift_iqr`는 false-alarm median과 true-negative median 차이를 true-negative IQR로 나눈 값이며, 인과적 feature importance가 아니다.
- block은 실제 공정조건 정답이 아니라 수집 순서 기반 proxy이므로 block 집중을 공정조건 효과로 해석하지 않는다.
- 이 분석도 이상이 포함된 window 탐지 결과에 대한 사후 분석이며 pre-failure 성능을 의미하지 않는다.
