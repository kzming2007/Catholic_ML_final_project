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
- `EXPERIMENT_LOG.md`: 실제 실행한 실험의 시간순 기록
- `outputs/`: 실행 결과 CSV/Markdown 저장 위치

## 현재 상태

- 기준일: 2026-08-10
- 완료: 데이터 감사, row-level baseline 재현, cycle_run 내부 window baseline, pre-failure baseline, 반복 split·threshold 검증, 공정조건 분리 가능성 진단, cycle_run 경계 재검증, acquisition block held-out 강건성 검증.
- 진행 중: block-held-out 결과를 주 결과로 고정하고 동일 분할에서 딥러닝 비교 범위를 설계하는 단계.
- 막힌 점: 실제 공정조건 분류는 cycle-to-condition 정답표가 없어 검증할 수 없다. 이 제약은 시계열 이상탐지 연구 진행을 막지는 않는다.

## 다음 작업

1. 구간 단위 이상탐지는 `08` block-held-out 결과를 주 결과로 사용한다.
2. pre-failure 결과(`03`-`05`)는 `GripLost`의 약한 사전 신호와 한계 분석으로 제한한다.
3. 딥러닝 비교는 `cycle_run` 경계와 동일 acquisition block held-out 분할을 고정한 뒤 수행한다.

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
```

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
- temperature 제거는 `System_Failure`에서 개선됐지만 target별 효과가 일관되지는 않아 ablation 결과를 함께 보고한다.

## 근거

- 코드 및 데이터: `dataset/ur3_cobotops.csv`, `research/common.py`, `research/00_data_audit.py`, `research/01_baseline_reproduction.py`, `research/02_window_feature_baseline.py`, `research/03_pre_failure_window_baseline.py`
- 결과물: `research/outputs/00_data_audit_report.md`, `research/outputs/01_baseline_results.md`, `research/outputs/02_window_feature_results.md`, `research/outputs/03_pre_failure_window_results.md`, `research/outputs/04_pre_failure_repeated_split_results.md`, `research/outputs/05_pre_failure_threshold_sensitivity_results.md`
- 공정조건 진단: `research/outputs/06_process_condition_separability.md`, `research/outputs/06_condition_cluster_summary.csv`, `research/outputs/06_condition_block_classification.csv`
- 경계·강건성 검증: `research/outputs/07_cycle_run_revalidation.md`, `research/outputs/08_block_held_out_robustness.md`
- 관련 메모: `research/2026-07-08_time_series_method_decision.md`

## 열린 질문

- 딥러닝 비교에서 10-step 공통 window를 쓸지, target별 대표 window를 쓸지 정해야 한다.
- `GripLost` pre-failure threshold 0.30을 후속 검증 후보로 유지할지 정해야 한다.
- 공정조건 분류는 대응표를 확보하지 않는 한 연구 범위에 포함하지 않는다.

## 갱신 기록

- 2026-07-10: 프로젝트 기준 문서로 지정하고, `03_pre_failure_window_baseline.py`까지 반영한 현재 상태를 정리했다.
- 2026-07-22: 반복 split 검증(`04`)과 threshold 민감도 검증(`05`) 결과를 반영했다.
- 2026-08-10: 공정조건 분리 가능성 진단(`06`)과 raw `cycle` ID 재등장 문제를 반영했다.
- 2026-08-10: `cycle_run` 기준으로 `02`-`05`를 재실행하고 경계 재검증(`07`)과 acquisition block held-out 검증(`08`)을 반영했다.
