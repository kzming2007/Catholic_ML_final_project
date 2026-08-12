# UR3 CobotOps 실험 일지

> [!note] 관리 원칙
> 실제로 실행한 실험만 시간순으로 누적한다. 입력되지 않은 조건이나 결과를 추정해서 채우지 않는다.

## 실험 기록

### 2026-07-08 - 00 데이터 감사

- 질문: 공개 `ur3_cobotops.csv`가 연구용 분류 실험에 어떤 구조와 제한을 갖는가?
- 데이터 및 입력: `dataset/ur3_cobotops.csv`
- 실행 조건: `python -X utf8 research\00_data_audit.py`
- 방법: 컬럼, 결측치, target 분포, timestamp 간격, cycle 구조, feature group을 점검했다.
- 결과: 원본 7,409 rows, 주요 결측 제거 후 모델 학습 기준 7,355 rows, unique cycle 240개를 확인했다. `System_Failure`는 `ProtectiveStop OR GripLost`로 정의했다.
- 해석: 공개 CSV의 timestamp는 대부분 약 1초 간격이며, `workload`, `movement speed`, `gripping force`는 직접 컬럼이 아니다.
- 한계 또는 오류: 일부 timestamp 간격이 길어 균일한 고주파 센서 데이터로 가정하면 안 된다.
- 다음 실험: row-level baseline과 cycle-aware split을 비교한다.
- 관련 파일: `research/outputs/00_data_audit_report.md`

### 2026-07-08 - 01 Row-level baseline 재현

- 질문: 기존 프로젝트의 분류 기반 baseline을 재현하고, random split과 cycle-group split 차이를 확인할 수 있는가?
- 데이터 및 입력: 원본 센서 feature 19개, power feature 6개, `Abs_Current_Sum`, targets `System_Failure`, `ProtectiveStop`, `GripLost`
- 실행 조건: `python -X utf8 research\01_baseline_reproduction.py`
- 방법: `Random Forest`, `SMOTE + Random Forest`를 row 단위 `random_stratified` split과 `cycle_group` split에서 비교했다.
- 결과: `cycle_group + rf_smote` 기준 Macro F1은 `System_Failure` 0.7901, `ProtectiveStop` 0.8428, `GripLost` 0.7623이었다.
- 해석: 기존 baseline은 재현되며, 고장 유형별 결과가 달라 `System_Failure` 통합 타깃만으로는 해석이 부족하다.
- 한계 또는 오류: row-level 입력은 시간 순서와 cycle 내부 변화를 직접 반영하지 않는다.
- 다음 실험: cycle 경계를 넘지 않는 window feature baseline을 만든다.
- 관련 파일: `research/outputs/01_baseline_results.md`

### 2026-07-08 - 02 Cycle 내부 window feature baseline

- 질문: cycle 내부의 짧은 시계열 구간을 feature로 요약하면 row-level baseline보다 분류 성능이 좋아지는가?
- 데이터 및 입력: cycle 경계를 넘지 않는 5, 10, 20 step window
- 실행 조건: `python -X utf8 research\02_window_feature_baseline.py`
- 방법: 각 window에서 mean, std, min, max, range, first-last delta, simple slope를 만들고 `cycle_group` split에서 `Random Forest`, `SMOTE + Random Forest`를 비교했다.
- 결과: `rf_smote` 기준 best Macro F1은 `System_Failure` 10-step 0.8593, `ProtectiveStop` 10-step 0.9323, `GripLost` 5-step 0.8796이었다.
- 해석: 구간 단위 이상탐지에서는 window feature가 row-level baseline보다 세 target 모두에서 개선됐다.
- 한계 또는 오류: window 안에 positive가 하나라도 있으면 positive로 두므로, 이미 발생한 이상 상태를 포함할 수 있다. 고장 발생 전 예측으로 해석하지 않는다.
- 다음 실험: first positive 이후 데이터를 제외한 pre-failure window baseline을 만든다.
- 관련 파일: `research/outputs/02_window_feature_results.md`, `research/outputs/02_row_vs_window_comparison.csv`

### 2026-07-09 - 03 First positive 이전 pre-failure window baseline

- 질문: 이미 positive가 된 이후 데이터를 제외하고도 고장 발생 직전 구간을 분류할 수 있는가?
- 데이터 및 입력: 각 cycle의 first positive step 이전 window, prediction horizon 3, 5, 10 step
- 실행 조건: `python -X utf8 research\03_pre_failure_window_baseline.py`
- 방법: first positive step 이후 또는 그 시점을 포함하는 window를 제외했다. window 종료 시점이 first positive step 이전 horizon 안에 있으면 positive로 두고, `cycle_group` split에서 `Random Forest`, `SMOTE + Random Forest`를 비교했다.
- 결과: `rf_smote` best 기준 `GripLost`는 window 5, horizon 3에서 positive F1 0.4375, `ProtectiveStop`은 window 10, horizon 3에서 positive F1 0.5000, `System_Failure`는 window 5, horizon 10에서 positive F1 0.0727이었다.
- 해석: pre-failure task는 구간 단위 이상탐지보다 훨씬 어렵다. `GripLost`와 `ProtectiveStop`은 일부 조합에서 약한 사전 신호가 보이나, `System_Failure`는 default threshold에서 거의 잡히지 않았다.
- 한계 또는 오류: positive window 수가 적고 단일 split 결과라 안정적인 결론으로 보기 어렵다. ROC-AUC나 PR-AUC가 높더라도 positive recall이 0에 가까운 조합은 실제 경고 모델로 해석하지 않는다.
- 다음 실험: 반복 cycle-group split 또는 group cross-validation으로 안정성을 확인하고, threshold 조정을 별도 실험으로 분리한다.
- 관련 파일: `research/outputs/03_pre_failure_window_results.md`, `research/outputs/03_pre_failure_window_best.csv`

### 2026-07-22 - 04 Pre-failure 반복 cycle-group split 검증

- 질문: `03` pre-failure 대표 조합의 결과가 단일 split 우연이 아니라 반복 split에서도 유지되는가?
- 데이터 및 입력: `System_Failure` 5-step/horizon 10, `ProtectiveStop` 10-step/horizon 3, `GripLost` 5-step/horizon 3
- 실행 조건: `python -X utf8 research\04_pre_failure_repeated_split_validation.py`
- 방법: target별 유효 `cycle_group` split 30회를 만들고 `SMOTE + Random Forest`를 반복 학습했다. default threshold 기준 positive recall, precision, F1, `runs_with_tp_rate`를 집계했다.
- 결과: `GripLost`는 30회 모두 positive를 1개 이상 잡았고 positive F1 평균은 0.3164였다. `ProtectiveStop`은 positive F1 평균 0.0656, `runs_with_tp_rate` 0.1로 단일 split 결과가 안정적이지 않았다. `System_Failure`는 positive F1 평균 0.0963으로 낮았다.
- 해석: pre-failure default threshold 기준으로는 `GripLost`가 가장 안정적인 개별 타깃이다. `ProtectiveStop`은 test positive 수가 평균 5개로 작아 결과 변동이 크고, `System_Failure`는 통합 타깃이라 사전 예측 성능이 낮다.
- 한계 또는 오류: 반복 split은 대표 조합만 대상으로 했으며, 전체 window/horizon 조합을 모두 반복 검증한 것은 아니다.
- 다음 실험: threshold를 낮춰 recall을 회수할 수 있는지 확인한다.
- 관련 파일: `research/outputs/04_pre_failure_repeated_split_results.md`, `research/outputs/04_pre_failure_repeated_split_summary.csv`

### 2026-07-22 - 05 Pre-failure threshold 민감도 검증

- 질문: default threshold 0.50에서 놓친 pre-failure positive를 threshold 조정으로 회수할 수 있는가?
- 데이터 및 입력: `04`와 같은 target별 대표 조합, threshold 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50
- 실행 조건: `python -X utf8 research\05_pre_failure_threshold_sensitivity.py`
- 방법: target별 유효 `cycle_group` split 30회에서 `SMOTE + Random Forest`를 학습하고, 같은 score에 여러 threshold를 적용했다. 평균 false positive rate가 0.10 이하인 후보 중 positive F1 평균이 가장 높은 threshold를 추천 후보로 정했다.
- 결과: 추천 후보는 `GripLost` threshold 0.30, `ProtectiveStop` threshold 0.15, `System_Failure` threshold 0.30이었다. `GripLost`는 threshold 0.30에서 positive recall 평균 0.4284, precision 평균 0.3723, positive F1 평균 0.3803, false positive rate 평균 0.0125였다. `System_Failure`는 threshold 0.30에서 recall은 0.2669로 올라갔지만 positive F1 평균은 0.2069에 그쳤다.
- 해석: threshold 조정은 `GripLost`에서 가장 해석 가능하며, `System_Failure`는 threshold를 낮춰도 통합 타깃의 어려움이 남는다. `ProtectiveStop`은 false positive rate는 낮지만 positive 수가 작아 안정적인 주장으로 쓰기 어렵다.
- 한계 또는 오류: 추천 threshold는 후속 검증 후보일 뿐이며, 최종 결론에는 false positive 부담과 반복 split 분산을 함께 보고해야 한다.
- 다음 실험: 딥러닝 비교 전, `GripLost` 중심 pre-failure 설정을 유지할지 결정한다.
- 관련 파일: `research/outputs/05_pre_failure_threshold_sensitivity_results.md`, `research/outputs/05_pre_failure_threshold_sensitivity_recommended.csv`

### 2026-08-10 - 06 공정조건 분리 가능성 진단

- 질문: 논문에 제시된 `movement speed`, `workload`, `gripping force`의 서로 다른 수준을 공개 센서 데이터만으로 cycle별 독립 구분할 수 있는가?
- 데이터 및 입력: `dataset/ur3_cobotops.csv`, `dataset/dataset_02052023.xlsx`, IF-FCM 논문 DOI `10.1007/978-3-031-63851-0_6`
- 실행 조건: `python -X utf8 research\06_process_condition_separability.py`
- 방법: 논문의 25회 반복 기술을 근거로 25-cycle 후보 블록을 구성했다. 비연속적으로 재등장한 동일 `cycle` ID는 시간순 연속 구간별 `cycle_run`으로 분리했다. first positive 이전 healthy prefix에서 speed/current/Tool current 요약을 만들고, 조건별 K-Means 3-cluster, GMM BIC 1-5 component, 후보 블록 분류를 비교했다.
- 결과: `movement_speed` Silhouette 0.2044, `workload` 0.4037, `gripping_force` 0.2728이었고 세 조건 모두 GMM BIC가 시험 범위에서 5 components를 선택했다. process sensor Logistic Regression의 9개 후보 블록 balanced accuracy는 0.2060(chance 0.1111), temperature-only Random Forest는 0.9620이었다.
- 해석: 센서에 일부 regime 차이는 있으나 논문의 세 수준과 일치하는 단순한 3개 구조는 확인되지 않았다. 후보 블록 정체성은 공정 sensor보다 온도와 수집 순서의 영향을 크게 받아 thermal/session drift 교란 가능성이 높다.
- 한계 또는 오류: 공개 Excel/CSV와 검색 가능한 부속자료에는 cycle-to-condition 정답표가 없다. 따라서 cluster를 실제 1/2/3 kg, 60/80/100%, 80/100/120 N으로 이름 붙이거나 supervised accuracy를 계산할 수 없다. healthy prefix 5 rows 미만 시행 71개를 제외해 선택 편향 가능성도 있다.
- 추가 발견: raw `cycle` ID 224, 225, 226, 227, 229, 230, 231, 232, 233이 서로 떨어진 시행 구간에 재등장한다. 기존 `02`-`05`가 raw `cycle`만으로 window/group을 만들었으므로 `cycle_run` 기준 재검증이 필요하다.
- 다음 실험: 기존 window/pre-failure 실험을 `cycle_run` 기준으로 다시 실행해 차이를 확인한다. 실제 공정조건 분류는 저자 또는 원 수집팀의 cycle-to-condition mapping 확보 후 진행한다.
- 관련 파일: `research/outputs/06_process_condition_separability.md`, `research/outputs/06_condition_cluster_summary.csv`, `research/outputs/06_condition_block_classification.csv`

### 2026-08-10 - 07 Cycle-run 경계 재검증

- 질문: 비연속적으로 재등장한 raw `cycle` ID를 분리하면 기존 `02`-`05` 결론이 유지되는가?
- 데이터 및 입력: 수정 전 요약 결과 `research/outputs/raw_cycle_reference/`, 수정 후 `cycle_run` 기준 `02`-`05` 결과
- 실행 조건: `python -X utf8 research\02_window_feature_baseline.py`, `03_pre_failure_window_baseline.py`, `04_pre_failure_repeated_split_validation.py`, `05_pre_failure_threshold_sensitivity.py`, `07_cycle_run_revalidation.py`
- 방법: 행 순서에서 `cycle` 값이 바뀔 때마다 `cycle_run`을 부여하고, window 생성·first positive 탐색·group split을 `cycle_run` 기준으로 다시 수행했다.
- 결과: 5/10/20-step에서 target별로 각각 24/54/104개의 경계-crossing window가 제거됐다. `02`와 `03`의 단일 split best 설정은 일부 바뀌었다. 동일 설정 30회 반복 positive F1 평균은 `GripLost` 0.3164 -> 0.3058, `ProtectiveStop` 0.0656 -> 0.0867, `System_Failure` 0.0963 -> 0.0920이었다.
- 해석: raw `cycle` 경계 문제는 수정이 필요했고 단일 best 수치는 불안정했다. 그러나 `GripLost`가 상대적으로 가장 안정적인 pre-failure 타깃이라는 반복 검증 결론은 유지됐다.
- threshold 결과: `GripLost`와 `System_Failure` 추천 후보는 0.30을 유지했고, `ProtectiveStop`은 0.15에서 0.20으로 바뀌었다. 수정 후 positive F1 평균은 각각 0.4101, 0.1856, 0.3231이었다.
- 한계 또는 오류: 수정 전후 단일 split은 test 시행 구성이 달라져 delta를 순수한 경계 효과로만 해석할 수 없다. 반복 split 비교를 주 근거로 사용한다.
- 다음 실험: acquisition block 전체를 제외하는 구간 단위 이상탐지 강건성 평가를 수행한다.
- 관련 파일: `research/outputs/07_cycle_run_revalidation.md`, `research/outputs/07_cycle_run_04_repeated_comparison.csv`, `research/outputs/07_cycle_run_05_threshold_comparison.csv`

### 2026-08-10 - 08 Acquisition block held-out 강건성 검증

- 질문: 특정 25-cycle acquisition block 전체를 학습에서 제외해도 구간 단위 이상탐지가 일반화되는가?
- 데이터 및 입력: `cycle_run` 경계의 5/10/20-step window, 20개 이상 cycle_run이 있는 후보 block 1, 2, 3, 4, 5, 7, 8, 9, 10
- 실행 조건: `python -X utf8 research\08_block_held_out_robustness.py`
- 방법: 한 block 전체를 test로 두고 나머지 8개 block으로 `SMOTE + Random Forest`를 학습했다. 전체 sensor와 temperature 제거 feature를 비교하고 9개 holdout의 평균·표준편차·최솟값을 집계했다.
- 결과: 모든 target/window/feature 설정에서 9개 block 모두 positive를 1개 이상 탐지했다. 대표 결과는 `System_Failure` 10-step/no-temperature Macro F1 0.8580, positive recall 0.7794, positive F1 0.7839, 최소 recall 0.4872였다. `ProtectiveStop` 10-step/all-sensors는 0.8650/0.7594/0.7620/0.2593, `GripLost` 20-step/no-temperature는 0.8759/0.7337/0.7911/0.3415였다.
- 해석: 구간 단위 이상탐지는 특정 acquisition block 하나에만 의존한 결과가 아니며, 공정조건 정답표 없이도 block/session 변화에 대한 일반화 성능을 평가할 수 있다.
- 한계 또는 오류: 후보 block은 실제 공정조건 라벨이 아니라 condition과 time/session drift가 섞인 proxy다. window 안에 이미 이상 상태가 포함될 수 있으므로 pre-failure 예측 결과로 해석하지 않는다.
- temperature ablation: `System_Failure`에서는 제거 효과가 비교적 일관됐지만 다른 target에서는 metric별 효과가 엇갈려, 온도를 일괄 제외하기보다 ablation으로 보고한다.
- 다음 실험: 동일 `cycle_run` 경계와 block-held-out 분할을 고정해 window feature와 sequence model을 비교한다.
- 관련 파일: `research/outputs/08_block_held_out_robustness.md`, `research/outputs/08_block_held_out_summary.csv`, `research/outputs/08_block_held_out_best.csv`

### 2026-08-12 - 09 공통 10-step 사건 단위 block-held-out 검증

- 질문: 딥러닝 비교 전에 10-step과 동일 후보 block 분할을 고정했을 때, Random Forest가 실제 이상 event cycle을 얼마나 탐지하고 정상 cycle에서 얼마나 오경보를 내는가?
- 데이터 및 입력: `cycle_run` 경계의 10-step window, 후보 block 1, 2, 3, 4, 5, 7, 8, 9, 10, 전체 sensor 및 temperature 제거 feature.
- 실행 조건: `python -X utf8 research\09_event_level_block_validation.py`, `SMOTE + Random Forest`, 300 trees, positive decision rule `score > 0.50`.
- 방법: 각 block을 한 번씩 test로 두고 나머지 8개 block만 학습했다. 실제 positive가 포함된 window를 하나 이상 맞힌 event cycle을 탐지로 계산하고, 정상 cycle에서 positive 예측이 하나라도 발생하면 cycle 오경보로 계산했다. 모든 window score와 cycle별 집계 결과를 저장했다.
- 결과: 전체 sensor 기준 `System_Failure`는 event cycle 87/87 탐지, 정상 cycle 오경보율 0.0522였다. `ProtectiveStop`은 59/62 탐지와 오경보율 0.0429, `GripLost`는 37/39 탐지와 오경보율 0.0429였다. 가장 어려운 block의 event recall은 각각 1.0, 0.5, 0.6667이었다.
- Temperature ablation: temperature 제거 시 `System_Failure` window positive F1은 0.7451에서 0.7839로 증가했지만 정상 cycle 오경보율도 0.0522에서 0.0783으로 증가했다. `ProtectiveStop`과 `GripLost`에서도 오경보율이 증가해 전체 sensor를 주 설정으로 유지한다.
- 정합성 검증: 54개 block 결과의 window-level Macro F1, positive recall, positive F1, PR-AUC가 기존 `08`의 동일 10-step 결과와 모두 일치했다. 저장된 24,210개 window 예측에서 summary를 재집계한 결과도 원본 summary와 일치했다.
- 해석: 구간 단위 이상탐지는 사건 단위에서도 높은 탐지율을 보이지만 `ProtectiveStop`과 `GripLost`는 특정 block에서 누락이 남는다. 이 결과는 이상이 포함된 window 탐지이며 고장 발생 전 예측으로 해석하지 않는다.
- 한계 또는 오류: 첫 실행은 120초 제한으로 종료되어 결과가 저장되지 않았다. 동일한 300-tree 설정으로 실행 시간을 늘려 134.6초에 완료했으며, 0.5 동률 처리 차이를 발견해 기존 `RandomForest.predict()`와 같도록 `score > 0.50`으로 저장 score를 재집계했다.
- 다음 실험: 전체 sensor, 10-step, 동일 9개 block, 동일 사건 단위 지표를 고정해 1D CNN과 LSTM을 비교한다.
- 관련 파일: `research/outputs/09_event_level_block_validation.md`, `research/outputs/09_event_level_block_summary.csv`, `research/outputs/09_event_level_cycle_results.csv`, `research/outputs/09_event_level_window_predictions.csv`

### 2026-08-12 - 10 고정 1D CNN/LSTM sequence model 비교

- 질문: 동일한 10-step 구간에서 step 순서를 직접 입력받는 소형 1D CNN과 LSTM이 통계적 window feature 기반 Random Forest보다 이상 event를 안정적으로 탐지하는가?
- 데이터 및 입력: `cycle_run` 경계를 넘지 않는 10-step×26-feature raw sequence, target `System_Failure`, `ProtectiveStop`, `GripLost`, 후보 block 1, 2, 3, 4, 5, 7, 8, 9, 10.
- 실행 조건: 데이터 준비와 집계는 프로젝트 Python 환경을 사용했다. 학습은 기존 `D:\Projects\Cube_Codex\.venvs\sam2_clean` 환경의 Python 3.12.3, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 3060 Ti를 패키지 변경 없이 재사용했다.
- 방법: 각 block을 한 번씩 test로 두고, 다음 순서의 outer-train block 1개를 validation으로 사용해 epoch를 선택했다. 선택된 epoch로 test block을 제외한 8개 block 전체를 재학습했다. 고정 소형 1D CNN과 단층 LSTM에 class-weighted BCE, Adam, threshold `score > 0.50`, seed 42/43/44를 적용했다.
- 실행 범위: 3 targets×2 models×9 test blocks×3 seeds의 162회 학습과 72,630개 window prediction을 완료했다. 각 target/model/seed 조합은 9개 test block과 4,035개 window를 모두 포함했고, test/validation block 분리와 threshold 규칙을 검증했다. 기록된 개별 학습 시간의 합은 486.7초였다.
- 결과: `System_Failure`의 event recall/정상 cycle 오경보율은 1D CNN 0.9732/0.1594, LSTM 0.9617/0.2058, Random Forest 1.0000/0.0522였다. `ProtectiveStop`은 각각 0.9355/0.2024, 0.9086/0.2119, 0.9516/0.0429였고, `GripLost`는 0.9487/0.3252, 0.9487/0.3763, 0.9487/0.0429였다.
- Window 결과: pooled positive F1 기준 `System_Failure`는 1D CNN 0.7097, LSTM 0.6749, Random Forest 0.7326이었다. `ProtectiveStop`은 0.6769, 0.6953, 0.7861이었고, `GripLost`는 0.5451, 0.5302, 0.6328이었다. PR-AUC도 세 target 모두 Random Forest가 가장 높았다.
- 해석: 소형 sequence model도 event cycle 대부분을 탐지했지만 정상 cycle에서 경보가 누적됐다. 현재 데이터 규모와 고정 block 평가에서는 짧은 시계열 구간을 통계량으로 정제한 Random Forest가 더 균형 잡힌 기준선이다. 딥러닝 사용 자체가 성능 향상을 보장하지 않는다는 비교 결과로 해석한다.
- 한계 또는 오류: window 안에 이미 positive 상태가 포함될 수 있어 pre-failure 예측 결과가 아니다. 후보 block은 실제 공정조건 라벨이 아니라 cycle 번호 기반 proxy다. Random Forest는 고정 1회, deep learning은 3개 seed 평균이며, architecture 탐색은 수행하지 않았다.
- 다음 실험: 추가 모델을 바로 늘리지 않고 false alarm이 집중된 정상 cycle과 성능이 낮은 held-out block을 먼저 분석할지 결정한다.
- 관련 파일: `research/10_prepare_sequence_data.py`, `research/10_torch_sequence_models.py`, `research/10_sequence_model_results.py`, `research/outputs/10_sequence_model_comparison.md`, `research/outputs/10_sequence_model_summary.csv`, `research/outputs/10_sequence_training_runs.csv`

## 공통 기준

- 재현에 필요한 경로, 스크립트 버전, 주요 파라미터를 남긴다.
- 실패한 실험도 삭제하지 않고 실패 원인과 다음 판단을 기록한다.
- 사람의 검토가 필요한 해석은 중앙 검토 대기열에 연결한다.
