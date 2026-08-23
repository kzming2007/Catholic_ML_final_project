# 13 최종 평가 지표표

## 목적

연구계획서에 명시한 confusion matrix, Accuracy, Macro F1, Precision, Recall, ROC-AUC, PR-AUC를 기존 `12` prediction에서 재집계한다. 새 모델 학습이나 threshold 선택은 수행하지 않는다.

## Window-level seed 평균

| target | model_variant | seed_count | tn_mean | fp_mean | fn_mean | tp_mean | accuracy_mean | macro_f1_mean | positive_precision_mean | positive_recall_mean | positive_f1_mean | roc_auc_mean | pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | rf_19_raw_window | 1 | 2871.0 | 190.0 | 299.0 | 675.0 | 0.8788 | 0.8278 | 0.7803 | 0.693 | 0.7341 | 0.9145 | 0.8137 |
| System_Failure | 1d_cnn_19_raw | 3 | 2678.0 | 383.0 | 306.0 | 668.0 | 0.8292 | 0.7723 | 0.6357 | 0.6858 | 0.6585 | 0.8634 | 0.6879 |
| System_Failure | lstm_autoencoder_19_raw_q95 | 3 | 3009.6667 | 51.3333 | 957.6667 | 16.3333 | 0.7499 | 0.4439 | 0.2418 | 0.0168 | 0.0314 | 0.4306 | 0.2228 |
| ProtectiveStop | rf_19_raw_window | 1 | 3375.0 | 97.0 | 137.0 | 426.0 | 0.942 | 0.8755 | 0.8145 | 0.7567 | 0.7845 | 0.9594 | 0.8557 |
| ProtectiveStop | 1d_cnn_19_raw | 3 | 3129.6667 | 342.3333 | 110.3333 | 452.6667 | 0.8878 | 0.8001 | 0.5722 | 0.804 | 0.6678 | 0.9132 | 0.7659 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | 3 | 3420.6667 | 51.3333 | 546.6667 | 16.3333 | 0.8518 | 0.4857 | 0.2418 | 0.029 | 0.0518 | 0.4603 | 0.1629 |
| GripLost | rf_19_raw_window | 1 | 3567.0 | 32.0 | 226.0 | 210.0 | 0.9361 | 0.7923 | 0.8678 | 0.4817 | 0.6195 | 0.9069 | 0.7267 |
| GripLost | 1d_cnn_19_raw | 3 | 3214.6667 | 384.3333 | 160.0 | 276.0 | 0.8651 | 0.7127 | 0.4181 | 0.633 | 0.5035 | 0.8731 | 0.5888 |
| GripLost | lstm_autoencoder_19_raw_q95 | 3 | 3531.3333 | 67.6667 | 436.0 | 0.0 | 0.8752 | 0.4667 | 0.0 | 0.0 | 0.0 | 0.4089 | 0.0834 |

- Random Forest는 고정 seed 1회이고 딥러닝 모델은 3개 seed 평균이다.
- `tn/fp/fn/tp_mean`은 딥러닝에서 seed별 confusion count의 평균이므로 정수형 단일 confusion matrix가 아니다. Seed별 원값은 `13_window_seed_metrics.csv`에 있다.
- ROC-AUC와 PR-AUC는 저장된 window score로 계산했다.

## Event-aware cycle consensus confusion matrix

| target | model_variant | seed_count | tn | fp | fn | tp | accuracy | macro_f1 | event_precision | event_recall | event_f1 | normal_cycle_false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | rf_19_raw_window | 1 | 110 | 5 | 2 | 85 | 0.9653 | 0.9648 | 0.9444 | 0.977 | 0.9605 | 0.0435 |
| System_Failure | 1d_cnn_19_raw | 3 | 93 | 22 | 2 | 85 | 0.8812 | 0.881 | 0.7944 | 0.977 | 0.8763 | 0.1913 |
| System_Failure | lstm_autoencoder_19_raw_q95 | 3 | 100 | 15 | 83 | 4 | 0.5149 | 0.3733 | 0.2105 | 0.046 | 0.0755 | 0.1304 |
| ProtectiveStop | rf_19_raw_window | 1 | 136 | 4 | 3 | 59 | 0.9653 | 0.9595 | 0.9365 | 0.9516 | 0.944 | 0.0286 |
| ProtectiveStop | 1d_cnn_19_raw | 3 | 113 | 27 | 4 | 58 | 0.8465 | 0.8342 | 0.6824 | 0.9355 | 0.7891 | 0.1929 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | 3 | 125 | 15 | 58 | 4 | 0.6386 | 0.4364 | 0.2105 | 0.0645 | 0.0988 | 0.1071 |
| GripLost | rf_19_raw_window | 1 | 157 | 6 | 3 | 36 | 0.9554 | 0.9305 | 0.8571 | 0.9231 | 0.8889 | 0.0368 |
| GripLost | 1d_cnn_19_raw | 3 | 123 | 40 | 3 | 36 | 0.7871 | 0.7386 | 0.4737 | 0.9231 | 0.6261 | 0.2454 |
| GripLost | lstm_autoencoder_19_raw_q95 | 3 | 143 | 20 | 39 | 0 | 0.7079 | 0.4145 | 0.0 | 0.0 | 0.0 | 0.1227 |

- Event cycle은 실제 positive window를 하나 이상 탐지해야 TP로 계산한다.
- Normal cycle은 어느 window에서든 경보가 발생하면 FP로 계산한다.
- 딥러닝은 3개 seed 중 2개 이상인 consensus, Random Forest는 고정 1회다.
- Cycle score aggregation을 사전 고정하지 않았으므로 cycle ROC-AUC와 PR-AUC는 사후 생성하지 않는다.

## 해석 범위

- Window-level과 cycle-level confusion matrix는 평가 단위가 다르므로 직접 합치지 않는다.
- 이 표는 기존 결과의 보고 지표를 보완하며 새로운 독립 실험 결과가 아니다.
- Primary 결론은 `12_matched_lstm_autoencoder_comparison.md`의 event recall과 정상 cycle 오경보율을 유지한다.
