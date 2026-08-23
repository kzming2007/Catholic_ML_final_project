# 12 동일 센서 입력 기반 LSTM Autoencoder 후속 비교

## 고정 설계

- 사전 고정 문서: `research/2026-08-23_lstm_autoencoder_preregistration.md`.
- 공통 입력: `cycle_run` 경계를 넘지 않는 10-step×19개 원본 센서.
- 모델: SMOTE+Random Forest window summary, supervised 1D CNN, normal-only LSTM Autoencoder.
- Outer 평가: 9개 candidate block을 한 번씩 test로 사용.
- Autoencoder primary threshold: calibration 정상 cycle 최대 reconstruction error의 95th percentile.
- Deep learning cycle consensus: 3개 seed 중 2개 이상.

## Primary consensus 결과

| target | model_variant | seed_count | event_cycle_count | detected_event_cycle_count | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycle_count | false_alarm_cycle_count | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | rf_19_raw_window | 1 | 87 | 85 | 0.977 | 0.92 | 0.9937 | 0.95 | 115 | 5 | 0.0435 | 0.0187 | 0.0978 | 0.1667 |
| ProtectiveStop | rf_19_raw_window | 1 | 62 | 59 | 0.9516 | 0.8671 | 0.9834 | 0.5 | 140 | 4 | 0.0286 | 0.0112 | 0.0712 | 0.1429 |
| GripLost | rf_19_raw_window | 1 | 39 | 36 | 0.9231 | 0.7968 | 0.9735 | 0.6667 | 163 | 6 | 0.0368 | 0.017 | 0.078 | 0.15 |
| System_Failure | 1d_cnn_19_raw | 3 | 87 | 85 | 0.977 | 0.92 | 0.9937 | 0.8 | 115 | 22 | 0.1913 | 0.1299 | 0.2727 | 0.8333 |
| ProtectiveStop | 1d_cnn_19_raw | 3 | 62 | 58 | 0.9355 | 0.8455 | 0.9746 | 0.5 | 140 | 27 | 0.1929 | 0.1361 | 0.2661 | 0.55 |
| GripLost | 1d_cnn_19_raw | 3 | 39 | 36 | 0.9231 | 0.7968 | 0.9735 | 0.5 | 163 | 40 | 0.2454 | 0.1857 | 0.3168 | 0.7143 |
| System_Failure | lstm_autoencoder_19_raw_q95 | 3 | 87 | 4 | 0.046 | 0.018 | 0.1123 | 0.0 | 115 | 15 | 0.1304 | 0.0807 | 0.2041 | 0.4 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | 3 | 62 | 4 | 0.0645 | 0.0254 | 0.1545 | 0.0 | 140 | 15 | 0.1071 | 0.066 | 0.1693 | 0.375 |
| GripLost | lstm_autoencoder_19_raw_q95 | 3 | 39 | 0 | 0.0 | -0.0 | 0.0897 | 0.0 | 163 | 20 | 0.1227 | 0.0809 | 0.1819 | 0.4 |

## Seed 평균 window 결과

| target | model_variant | seeds | event_cycle_recall_mean | normal_cycle_false_alarm_rate_mean | window_macro_f1_mean | window_positive_f1_mean | window_pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | rf_19_raw_window | 1 | 0.977 | 0.0435 | 0.8278 | 0.7341 | 0.8137 |
| ProtectiveStop | rf_19_raw_window | 1 | 0.9516 | 0.0286 | 0.8755 | 0.7845 | 0.8557 |
| GripLost | rf_19_raw_window | 1 | 0.9231 | 0.0368 | 0.7923 | 0.6195 | 0.7267 |
| System_Failure | 1d_cnn_19_raw | 3 | 0.9464 | 0.1913 | 0.7723 | 0.6585 | 0.6879 |
| ProtectiveStop | 1d_cnn_19_raw | 3 | 0.9355 | 0.2 | 0.8001 | 0.6678 | 0.7659 |
| GripLost | 1d_cnn_19_raw | 3 | 0.9145 | 0.272 | 0.7127 | 0.5035 | 0.5888 |
| System_Failure | lstm_autoencoder_19_raw_q95 | 3 | 0.046 | 0.1275 | 0.4439 | 0.0314 | 0.2228 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | 3 | 0.0645 | 0.1048 | 0.4857 | 0.0518 | 0.1629 |
| GripLost | lstm_autoencoder_19_raw_q95 | 3 | 0.0 | 0.1207 | 0.4667 | 0.0 | 0.0834 |

## Autoencoder threshold 민감도

| target | model_variant | seed_count | event_cycle_count | detected_event_cycle_count | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycle_count | false_alarm_cycle_count | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | lstm_autoencoder_19_raw_q90 | 3 | 87 | 7 | 0.0805 | 0.0395 | 0.1569 | 0.0 | 115 | 24 | 0.2087 | 0.1444 | 0.2918 | 0.4667 |
| System_Failure | lstm_autoencoder_19_raw_q95 | 3 | 87 | 4 | 0.046 | 0.018 | 0.1123 | 0.0 | 115 | 15 | 0.1304 | 0.0807 | 0.2041 | 0.4 |
| System_Failure | lstm_autoencoder_19_raw_q97.5 | 3 | 87 | 4 | 0.046 | 0.018 | 0.1123 | 0.0 | 115 | 15 | 0.1304 | 0.0807 | 0.2041 | 0.4 |
| ProtectiveStop | lstm_autoencoder_19_raw_q90 | 3 | 62 | 7 | 0.1129 | 0.0558 | 0.2152 | 0.0 | 140 | 24 | 0.1714 | 0.118 | 0.2424 | 0.4375 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | 3 | 62 | 4 | 0.0645 | 0.0254 | 0.1545 | 0.0 | 140 | 15 | 0.1071 | 0.066 | 0.1693 | 0.375 |
| ProtectiveStop | lstm_autoencoder_19_raw_q97.5 | 3 | 62 | 4 | 0.0645 | 0.0254 | 0.1545 | 0.0 | 140 | 15 | 0.1071 | 0.066 | 0.1693 | 0.375 |
| GripLost | lstm_autoencoder_19_raw_q90 | 3 | 39 | 0 | 0.0 | -0.0 | 0.0897 | 0.0 | 163 | 32 | 0.1963 | 0.1426 | 0.264 | 0.5 |
| GripLost | lstm_autoencoder_19_raw_q95 | 3 | 39 | 0 | 0.0 | -0.0 | 0.0897 | 0.0 | 163 | 20 | 0.1227 | 0.0809 | 0.1819 | 0.4 |
| GripLost | lstm_autoencoder_19_raw_q97.5 | 3 | 39 | 0 | 0.0 | -0.0 | 0.0897 | 0.0 | 163 | 20 | 0.1227 | 0.0809 | 0.1819 | 0.4 |

## Random Forest와 cycle 오류 겹침

| target | comparison_model | error_type | eligible_cycles | rf_error_count | comparison_error_count | shared_error_count | comparison_only_error_count | rf_only_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 1d_cnn_19_raw | false_alarm | 115 | 5 | 22 | 2 | 20 | 3 |
| System_Failure | 1d_cnn_19_raw | event_miss | 87 | 2 | 2 | 0 | 2 | 2 |
| System_Failure | lstm_autoencoder_19_raw_q95 | false_alarm | 115 | 5 | 15 | 0 | 15 | 5 |
| System_Failure | lstm_autoencoder_19_raw_q95 | event_miss | 87 | 2 | 83 | 2 | 81 | 0 |
| ProtectiveStop | 1d_cnn_19_raw | false_alarm | 140 | 4 | 27 | 2 | 25 | 2 |
| ProtectiveStop | 1d_cnn_19_raw | event_miss | 62 | 3 | 4 | 2 | 2 | 1 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | false_alarm | 140 | 4 | 15 | 0 | 15 | 4 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | event_miss | 62 | 3 | 58 | 3 | 55 | 0 |
| GripLost | 1d_cnn_19_raw | false_alarm | 163 | 6 | 40 | 3 | 37 | 3 |
| GripLost | 1d_cnn_19_raw | event_miss | 39 | 3 | 3 | 1 | 2 | 2 |
| GripLost | lstm_autoencoder_19_raw_q95 | false_alarm | 163 | 6 | 20 | 0 | 20 | 6 |
| GripLost | lstm_autoencoder_19_raw_q95 | event_miss | 39 | 3 | 39 | 3 | 36 | 0 |

## 사전 고정 해석 규칙 적용

- `System_Failure` RF 대비 1D CNN: Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다.
- `System_Failure` RF 대비 LSTM Autoencoder q95: Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다.
- `ProtectiveStop` RF 대비 1D CNN: Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다.
- `ProtectiveStop` RF 대비 LSTM Autoencoder q95: Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다.
- `GripLost` RF 대비 1D CNN: Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다.
- `GripLost` RF 대비 LSTM Autoencoder q95: Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다.
- 위 우세 판단은 recall과 오경보의 방향만 본 기술적 Pareto 비교이며 통계적 유의성을 뜻하지 않는다.

## 실행 범위

- Random Forest run: 27개.
- 1D CNN run: 81개.
- LSTM Autoencoder run: 27개.
- Autoencoder calibration 정상 cycle 수 범위: 13-36개.
- PyTorch: 2.7.1+cu128, CUDA: 12.8, device: cuda.
- 기록된 전체 학습 시간: 395.0초.

## 해석 제한

- Window 4,035개는 겹치므로 독립 표본 수가 아니다. Cycle-level n은 202개다.
- Wilson interval은 cycle 비율의 불확실성을 나타내지만 block/session 상관을 제거하지 않는다.
- 9개 block은 실제 공정조건 정답이 아니라 수집 구간 proxy다.
- 동일 데이터의 기존 결과를 이미 확인했으므로 독립 외부 검증이 아니라 사전 고정한 내부 후속 비교다.
- Autoencoder sensitivity threshold 중 가장 좋은 값을 primary 결과로 교체하지 않는다.
