# UR3 CobotOps 연구 작업 공간

> [!note] 문서 역할
> 이 문서는 ML 로봇팔 연구의 프로젝트 기준 문서로 사용한다. 상태가 달라졌을 때만 갱신하고, 실험 세부 결과는 `EXPERIMENT_LOG.md`와 `outputs/` 아래 결과 파일에 남긴다.

이 폴더는 기말 프로젝트 원본 노트북을 직접 수정하지 않고, 학술연구장학 연구 실행을 위해 만든 재현용 작업 공간이다.

## 목적

- 기존 UR3 CobotOps 기말 프로젝트의 분류 기반 이상탐지 baseline을 재현한다.
- row 단위 분류의 한계를 확인하고, cycle-aware split과 cycle 내부 window feature로 시계열 맥락을 반영한다.
- 구간 단위 이상탐지와 고장 발생 전 예측 실험을 분리하여 해석한다.
- 공개 센서만으로 논문의 공정조건 수준을 cycle별로 구분할 수 있는지 증거 범위를 진단한다.

## 확정 범위

- 포함: 공개 `ur3_cobotops.csv`, `System_Failure`, `ProtectiveStop`, `GripLost`, `Random Forest`, `SMOTE + Random Forest`, cycle-aware split, window feature baseline.
- 포함: 고정 10-step raw sequence를 입력으로 사용하는 소형 1D CNN과 단층 LSTM 비교.
- 제외: 공개 CSV에 없는 `workload`, `movement speed`, `gripping force`를 직접 feature처럼 해석하는 분석, 실제 로봇팔 제어 시스템 구현, 원본 데이터의 Obsidian Vault 복사.

## 현재 파일

- `common.py`: 데이터 로딩, 컬럼 정리, 타깃 생성, 공통 feature 정의
- `00_data_audit.py`: 데이터셋 구조, 결측치, 타깃 분포, Timestamp/cycle 구조 점검
- `01_baseline_reproduction.py`: row 단위 baseline과 split 방식 비교
- `02_window_feature_baseline.py`: cycle_run 내부 window feature 기반 분류 baseline
- `03_pre_failure_window_baseline.py`: first positive 이후 데이터를 제외한 고장 발생 전 window 분류 baseline
- `04_pre_failure_repeated_split_validation.py`: pre-failure 대표 조합의 반복 cycle-run-group split 검증
- `05_pre_failure_threshold_sensitivity.py`: pre-failure 대표 조합의 threshold 민감도 검증
- `06_process_condition_separability.py`: 공정조건 3개 수준의 latent cluster 및 25-cycle 후보 블록 분리 가능성 진단
- `07_cycle_run_revalidation.py`: raw cycle 경계 결과와 cycle_run 수정 결과 비교
- `08_block_held_out_robustness.py`: 25-cycle acquisition block held-out 구간 이상탐지 강건성 검증
- `09_event_level_block_validation.py`: 공통 10-step·동일 block 분할의 사건 단위 탐지율과 정상 cycle 오경보 검증
- `10_prepare_sequence_data.py`: 기존 `09`와 동일한 10-step window를 10×26 raw sequence로 준비하고 정합성 검증
- `10_torch_sequence_models.py`: 외부 PyTorch 환경에서 고정 1D CNN/LSTM을 block-held-out 방식으로 학습
- `10_sequence_model_results.py`: Random Forest와 sequence model의 window·event·오경보 지표 비교
- `11_sequence_error_analysis.py`: 기존 예측에서 seed 반복 오류, block 집중도, 모델 간 오류 겹침, 정상 cycle 센서 차이 분석
- `2026-08-23_project_to_research_synthesis.md`: 수업 프로젝트와 학술연구의 차이, 현재 결론, 한계, 계획서 이행 상태를 정리한 종합 문서
- `2026-08-23_lstm_autoencoder_preregistration.md`: 동일 19개 센서 비교와 정상-only LSTM Autoencoder의 분석 기준을 결과 확인 전에 고정한 문서
- `12_matched_rf_baseline.py`: 10-step×19개 원본 센서를 133개 통계 feature로 요약한 고정 Random Forest 기준선
- `12_matched_torch_models.py`: 같은 10-step×19개 원본 센서를 사용하는 1D CNN과 정상-only LSTM Autoencoder 학습
- `12_matched_results.py`: 3-seed cycle consensus, Wilson 95% 신뢰구간, threshold 민감도와 모델 간 오류 비교 집계
- `13_final_evaluation_tables.py`: 저장된 `12` prediction에서 최종 confusion matrix와 ROC-AUC·PR-AUC 표를 재집계
- `2026-08-23_sensor_group_ablation_preregistration.md`: 센서 그룹 ablation의 비교 조건과 해석 기준을 결과 확인 전에 고정한 문서
- `14_sensor_group_ablation.py`: 동일 Random Forest 조건에서 센서 그룹 10개 변형을 비교하는 실행 코드
- `14_sensor_group_ablation_results.py`: Window·cycle·block 지표와 paired error를 집계하는 코드
- `requirements.txt`: 연구 스크립트를 실제 실행한 Python 패키지 버전
- `EXPERIMENT_LOG.md`: 실제 실행한 실험의 시간순 기록
- `outputs/`: 실행 결과 CSV/Markdown 저장 위치

## 현재 상태

- 기준일: 2026-08-23
- 완료: 데이터 감사, row-level baseline 재현, cycle_run 내부 window baseline, pre-failure baseline, 반복 split·threshold 검증, 공정조건 분리 가능성 진단, cycle_run 경계 재검증, acquisition block held-out 강건성 검증, 공통 10-step 사건 단위 기준선 검증, 고정 1D CNN/LSTM 비교, sequence model 오류 분석, 동일 19개 센서 기반 Random Forest·1D CNN·정상-only LSTM Autoencoder 비교, 최종 평가표 재집계, 사전 고정 센서 그룹 ablation.
- 현재 결론: 주 연구는 supervised 구간 단위 이상탐지로 둔다. 동일 19개 센서 비교에서 Random Forest의 event cycle recall/정상 cycle 오경보율은 `System_Failure` 0.9770/0.0435, `ProtectiveStop` 0.9516/0.0286, `GripLost` 0.9231/0.0368이었다. 1D CNN은 recall 0.9231~0.9770을 유지했지만 오경보율이 0.1913~0.2454로 증가했다. 정상-only LSTM Autoencoder q95는 recall 0~0.0645와 오경보율 0.1071~0.1304로 세 타깃 모두에서 기준선보다 열세였다. 센서 그룹 ablation에서는 전류 계열 제거가 `System_Failure`와 특히 `GripLost`를 악화시켰지만 `ProtectiveStop`에서는 recall과 오경보의 tradeoff가 나타났다. `Tool_current` 제거는 `System_Failure`와 `ProtectiveStop`에서 오히려 개선됐고 `GripLost`에서는 소폭 악화됐다. 따라서 관절 전류 계열의 예측 기여는 확인하되, 전류 전체가 항상 또는 유일하게 핵심이라는 주장은 하지 않는다.
- 막힌 점: 실제 공정조건 분류는 cycle-to-condition 정답표가 없어 검증할 수 없다. 이 제약은 시계열 이상탐지 연구 진행을 막지는 않는다.

## 다음 작업

1. `2026-08-23_project_to_research_synthesis.md`와 `13`·`14` 결과를 기준으로 공동연구자와 주 결과·비교 결과·부가 결과의 위계를 확정한다.
2. 최종 연구 서술은 동일 19개 센서 Random Forest를 공통 기준선으로 유지하고, 센서 그룹 ablation은 해석 보조 결과로 둔다.
3. 1D CNN과 정상-only LSTM Autoencoder는 딥러닝 비교군으로 보고하되, 성능 열세와 정상 분포 가정의 실패를 그대로 해석한다.
4. Pre-failure 결과(`03`-`05`)는 `GripLost`의 약한 사전 신호를 확인한 제한적 탐색 결과로 분리한다.
5. 이번 ablation 결과를 본 뒤 `drop_tool_current`를 새 주 모델로 교체하거나 threshold를 조정하지 않는다.
6. Logistic Regression·SVM을 동일 조건에서 재검증할지는 계획서 이행에 필요한 최소 비교인지 공동연구자와 결정한다.

## 실행 순서

```powershell
python -X utf8 research\00_data_audit.py
python -X utf8 research\01_baseline_reproduction.py
python -X utf8 research\02_window_feature_baseline.py
python -X utf8 research\03_pre_failure_window_baseline.py
python -X utf8 research\04_pre_failure_repeated_split_validation.py
python -X utf8 research\05_pre_failure_threshold_sensitivity.py
python -X utf8 research\06_process_condition_separability.py
python -X utf8 research\07_cycle_run_revalidation.py
python -X utf8 research\08_block_held_out_robustness.py
python -X utf8 research\09_event_level_block_validation.py
python -X utf8 research\10_prepare_sequence_data.py
& "D:\Projects\Cube_Codex\.venvs\sam2_clean\Scripts\python.exe" -X utf8 research\10_torch_sequence_models.py --manifest research\.sequence_cache\10_sequence_manifest.json --output-dir research\outputs --device cuda
python -X utf8 research\10_sequence_model_results.py
python -X utf8 research\11_sequence_error_analysis.py
python -X utf8 research\12_matched_rf_baseline.py
& "D:\Projects\Cube_Codex\.venvs\sam2_clean\Scripts\python.exe" -X utf8 research\12_matched_torch_models.py --manifest research\.sequence_cache\10_sequence_manifest.json --output-dir research\outputs --device cuda
python -X utf8 research\12_matched_results.py
python -X utf8 research\13_final_evaluation_tables.py
python -X utf8 research\14_sensor_group_ablation.py
python -X utf8 research\14_sensor_group_ablation_results.py
```

## 데이터 provenance 및 실행 환경

- 공식 출처: UCI Machine Learning Repository, UR3 CobotOps, DOI `10.24432/C5J891`, 데이터 라이선스 `CC BY 4.0`.
- `dataset/ur3_cobotops.csv` SHA-256: `C789CDA10ACB354A7C1689F617D94A5F39A93FD8CB6C004AD16D36CEA55A74A3`.
- `dataset/dataset_02052023.xlsx` SHA-256: `F0F10917DE9056908A82CEC1AA459DDF9DA2D2DB70D269B128F99241C8796091`.
- CSV와 Excel은 7,409행 × 24열 구조가 같고, 수치 차이는 최대 약 `4.4e-8`로 반올림 수준이다.
- IF-FCM 참고 논문 DOI: `10.1007/978-3-031-63851-0_6`.
- `09` 검증 환경: Python 3.12.3, NumPy 2.4.3, pandas 3.0.3, scikit-learn 1.8.0, imbalanced-learn 0.14.1, openpyxl 3.1.5.
- `10` sequence model 환경: `D:\Projects\Cube_Codex\.venvs\sam2_clean`, Python 3.12.3, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 3060 Ti. 기존 환경의 패키지는 변경하지 않았다.
- `research/requirements.txt`는 데이터 준비와 결과 집계 환경을 기록하며 PyTorch는 위 외부 환경에서 재사용한다.
- 논문의 RTDE 125 Hz 설명은 인터페이스 동작 주파수이며, 공개 CSV의 양수 Timestamp 간격 중앙값은 약 1.005초다. Window 크기는 초가 아니라 step으로 해석한다.

## 현재 연구상 주의점

- 논문에서 말하는 `workload`, `movement speed`, `gripping force`는 공개 CSV의 직접 컬럼이 아니다.
- 원본 `cycle` ID 224, 225, 226, 227, 229, 230, 231, 232, 233은 시간상 떨어진 두 시행 구간에 재등장한다. 현재 `02`-`05`는 `cycle_run` 기준으로 수정·재실행됐다.
- 공정조건별 3-cluster를 실제 1/2/3 kg, 60/80/100%, 80/100/120 N 정답 라벨로 간주하면 안 된다.
- 25-cycle 후보 블록은 온도 feature만으로 매우 잘 구분되어, 공정조건과 thermal/session drift가 혼재할 가능성이 크다.
- 따라서 본 연구는 해당 공정 조건을 직접 검증하기보다, 센서 시계열에 남아 있는 이상탐지 패턴을 분석한다.
- `System_Failure`는 `ProtectiveStop OR GripLost`로 정의하되, 두 고장 유형은 별도 타깃으로도 비교한다.
- row 단위 random split은 성능을 낙관적으로 보일 수 있으므로, `cycle_run_group` 및 acquisition block held-out 결과와 비교한다.
- 시계열 맥락은 `cycle_run` 경계를 넘지 않는 5, 10, 20 step window feature로 반영한다.
- `02`는 구간 단위 이상탐지이고, `03`은 first positive 이전 데이터만 쓰는 고장 발생 전 예측 실험이므로 성능을 직접 비교하지 않는다.
- `04` 반복 split 기준으로 `GripLost`는 positive를 잡는 결과가 상대적으로 안정적이고, `ProtectiveStop`과 `System_Failure`는 해석에 더 주의가 필요하다.
- `05` threshold 조정은 recall 개선 가능성을 보지만, false positive 부담이 함께 증가하므로 최종 성능 주장으로 바로 쓰지 않는다.
- `08`은 구간 단위 이상탐지에서 9개 acquisition block 모두 positive를 잡았지만, 이미 이상이 포함된 window 탐지이므로 pre-failure 결과로 해석하지 않는다.
- `09`의 event cycle 탐지는 실제 positive가 포함된 window에서의 탐지 여부이며, 고장 전 경고 성능이 아니다.
- `09`의 정상 cycle 오경보율은 겹치는 window의 false positive가 cycle 단위 경보로 누적되는 운영상 부담을 나타낸다.
- temperature 제거는 `System_Failure`에서 개선됐지만 target별 효과가 일관되지는 않아 ablation 결과를 함께 보고한다.
- `10`의 Random Forest는 10-step을 182개 통계 feature로 요약하고, 1D CNN/LSTM은 같은 구간을 10×26 순서 입력으로 사용한다.
- `10`의 deep learning 결과는 seed 3개 평균이며, Random Forest는 `09`의 고정 1회 prediction을 재사용한다.
- `10`의 window 지표는 seed별 9개 held-out prediction을 합친 pooled 지표라서, `09` 보고서의 block별 평균과 숫자가 직접 같지는 않다.
- `11`의 deep learning cycle consensus는 3개 seed 중 2개 이상이 같은 cycle에서 경보한 경우다. Sensor shift는 동일 window에 대한 2/3 consensus만 사용한다.
- `11`에서 deep learning 오경보의 대부분은 Random Forest와 겹치지 않았다. `System_Failure`의 deep-only 오경보는 1D CNN/LSTM 15/19개, `ProtectiveStop` 23/26개, `GripLost` 47/54개였다.
- 정상 cycle의 반복 오경보 window는 target별로 낮은 관절 움직임 범위, 특정 전류 수준, Tool current 변화 또는 온도 수준 차이를 보였으나, 겹치는 window의 기술통계이므로 인과적 feature importance로 해석하지 않는다.
- `12`는 모델 입력을 10-step×19개 원본 센서로 맞췄다. Random Forest만 각 센서의 mean, std, min, max, range, delta, slope를 계산해 133개 feature로 변환하며, 1D CNN과 LSTM Autoencoder는 같은 원본 순서를 직접 입력받는다.
- `12`의 Autoencoder는 `System_Failure=0`인 완전 정상 cycle만 학습했다. 정상 재구성 오차가 모든 고장 유형을 포괄하는 이상 점수라는 가정은 결과적으로 지지되지 않았다.
- Autoencoder calibration 정상 cycle은 outer fold별 13~36개로 적다. q90, q95, q97.5 민감도에서도 결론이 바뀌지 않았으며, 더 유리한 threshold를 주 결과로 교체하지 않는다.
- `12`의 cycle-level 표본 수는 전체 202개다. Wilson 신뢰구간은 비율의 불확실성만 나타내며 block/session 내부 상관이나 동일 데이터 재사용 문제를 해소하지 않는다.
- `13`은 저장된 `12` prediction을 최종 표 형식으로 재집계한 결과이며 독립적인 새 실험이 아니다.
- `14`의 `all_19`는 `12` Random Forest의 score와 prediction에 정확히 일치한다.
- `14`의 센서 제거·단독 사용은 feature 차원과 SMOTE 공간도 함께 바꾸므로 센서의 인과적 효과를 증명하지 않는다.
- `Tool_current`는 세 타깃에서 일관되게 유용하지 않았고, 온도 단독 탐지력은 낮지만 온도 제거 후 일부 정상 cycle 오경보가 증가했다.

## 근거

- 코드 및 데이터: `dataset/ur3_cobotops.csv`, `research/common.py`, `research/00_data_audit.py`, `research/01_baseline_reproduction.py`, `research/02_window_feature_baseline.py`, `research/03_pre_failure_window_baseline.py`
- 결과물: `research/outputs/00_data_audit_report.md`, `research/outputs/01_baseline_results.md`, `research/outputs/02_window_feature_results.md`, `research/outputs/03_pre_failure_window_results.md`, `research/outputs/04_pre_failure_repeated_split_results.md`, `research/outputs/05_pre_failure_threshold_sensitivity_results.md`
- 공정조건 진단: `research/outputs/06_process_condition_separability.md`, `research/outputs/06_condition_cluster_summary.csv`, `research/outputs/06_condition_block_classification.csv`
- 경계·강건성 검증: `research/outputs/07_cycle_run_revalidation.md`, `research/outputs/08_block_held_out_robustness.md`, `research/outputs/09_event_level_block_validation.md`
- Sequence model 비교: `research/outputs/10_sequence_model_comparison.md`, `research/outputs/10_sequence_model_summary.csv`, `research/outputs/10_sequence_block_results.csv`, `research/outputs/10_sequence_window_predictions.csv`
- 오류 분석: `research/outputs/11_sequence_error_analysis.md`, `research/outputs/11_error_cycle_details.csv`, `research/outputs/11_error_block_summary.csv`, `research/outputs/11_false_alarm_sensor_shifts.csv`
- 동일 센서 후속 비교: `research/2026-08-23_lstm_autoencoder_preregistration.md`, `research/outputs/12_matched_lstm_autoencoder_comparison.md`, `research/outputs/12_matched_consensus_summary.csv`, `research/outputs/12_matched_pairwise_cycle_errors.csv`
- 최종 평가표: `research/outputs/13_final_evaluation_tables.md`, `research/outputs/13_cycle_consensus_confusion_metrics.csv`
- 센서 그룹 ablation: `research/2026-08-23_sensor_group_ablation_preregistration.md`, `research/outputs/14_sensor_group_ablation.md`, `research/outputs/14_sensor_group_ablation_summary.csv`, `research/outputs/14_sensor_group_ablation_paired_errors.csv`
- 연구 종합: `research/2026-08-23_project_to_research_synthesis.md`
- 관련 메모: `research/2026-07-08_time_series_method_decision.md`

## 열린 질문

- 최종 보고서에서 supervised 구간 단위 이상탐지를 주 연구 질문으로 두고, 정상-only Autoencoder의 부정적 결과를 어느 수준까지 설명할지 공동연구자와 합의해야 한다.
- pre-failure를 `GripLost` 중심의 제한적 부가 분석으로 둘지 합의해야 한다.
- `GripLost` pre-failure threshold 0.30을 후속 검증 후보로 유지할지 정해야 한다.
- 공정조건 분류는 대응표를 확보하지 않는 한 연구 범위에 포함하지 않는다.
- 최종 보고서에서는 `all_19`를 공통 기준선으로 유지하고, 센서 그룹 ablation은 예측 기여를 제한적으로 해석하는 보조 결과로 둘지 합의해야 한다.
- 기존 Logistic Regression·SVM을 동일 19-sensor·9-block 조건으로 다시 실행할지는 필수 결론과 추가 비용을 비교해 결정해야 한다.

## 갱신 기록

- 2026-07-10: 프로젝트 기준 문서로 지정하고, `03_pre_failure_window_baseline.py`까지 반영한 현재 상태를 정리했다.
- 2026-07-22: 반복 split 검증(`04`)과 threshold 민감도 검증(`05`) 결과를 반영했다.
- 2026-08-10: 공정조건 분리 가능성 진단(`06`)과 raw `cycle` ID 재등장 문제를 반영했다.
- 2026-08-10: `cycle_run` 기준으로 `02`-`05`를 재실행하고 경계 재검증(`07`)과 acquisition block held-out 검증(`08`)을 반영했다.
- 2026-08-12: 공통 10-step 사건 단위 검증(`09`)을 반영하고 데이터 provenance와 실행 환경을 기록했다.
- 2026-08-12: 외부 PyTorch 환경을 변경 없이 재사용해 고정 1D CNN/LSTM 비교(`10`)를 완료하고 결과와 한계를 반영했다.
- 2026-08-23: 새 학습 없이 `09`·`10` 예측을 재사용해 seed 반복 오류, block 집중도, 모델 간 오류 겹침과 정상 cycle 센서 차이 분석(`11`)을 완료했다.
- 2026-08-23: 결과 확인 전에 동일 19개 센서 비교를 사전등록하고, Random Forest·1D CNN·정상-only LSTM Autoencoder의 9-block·3-seed 후속 비교(`12`)를 완료했다.
- 2026-08-23: 수업 프로젝트의 성과와 과장 가능성, 학술연구에서 추가한 검증, 현재 한계와 최종 보고서 구조를 연구 종합 문서로 정리했다.
- 2026-08-23: 저장된 prediction에서 최종 confusion matrix와 ROC-AUC·PR-AUC 표(`13`)를 재집계했다.
- 2026-08-23: 결과 확인 전에 센서 그룹 비교를 사전등록하고, 동일 Random Forest 조건의 10개 입력 변형·270회 block 평가(`14`)를 완료했다.
