# 2026-07-08 시계열 방법론 검토 메모

## 결정문

**시간상 연속한 cycle_run은 데이터 누수 방지를 위한 평가 단위로 쓰고, 시계열 맥락은 cycle_run 내부의 고정 길이 window와 rolling feature로 반영한다.**

원본 `cycle` ID는 메타데이터로 보존한다. 동일 ID가 시간상 떨어진 시행에 재등장하므로 실제 window 경계와 train/test group에는 행 순서상 연속 구간인 `cycle_run`을 사용한다.

## 데이터셋 기본 요소

- 원본 데이터는 7,409 rows이며, 주요 센서 결측과 `Robot_ProtectiveStop` 결측을 제외하면 모델 학습에 쓰는 행은 7,355 rows다.
- 센서 feature는 6개 joint의 current, temperature, speed와 `Tool_current`로 구성된다.
- 현재 연구용 feature는 원본 센서 feature 19개에 `Current x Speed` 기반 power feature 6개, `Abs_Current_Sum`을 추가한 26개 feature다.
- `Timestamp`는 대부분 약 1초 간격이지만, 일부 cycle 내부에 긴 간격이 있으므로 균일한 고주파 센서 데이터처럼 가정하지 않는다.
- `cycle`은 1부터 264까지 존재하나 실제 unique cycle은 240개다.
- 시간상 연속 구간으로 다시 나누면 `cycle_run`은 249개다. 원본 ID 224, 225, 226, 227, 229, 230, 231, 232, 233이 두 구간에 재등장한다.
- cycle 길이는 평균 약 30.9 step, 중앙값 25 step이며, 최소 11 step부터 최대 102 step까지 차이가 있다.
- `System_Failure`는 `ProtectiveStop OR GripLost`로 정의한다.
- point 기준 positive row는 `System_Failure` 518개, `ProtectiveStop` 278개, `GripLost` 243개다.
- cycle 기준 positive cycle은 `System_Failure` 107개, `ProtectiveStop` 78개, `GripLost` 51개다.
- `ProtectiveStop` positive는 cycle 앞쪽에 더 많이 나타나고, `GripLost` positive는 cycle 뒤쪽에 더 많이 나타나는 경향이 있다.

## 방법론 선택

### 1. Row-level baseline 유지

기존 프로젝트의 기본 비교 기준으로 유지한다.

- 입력: 각 row의 센서값과 파생 feature
- 모델: `Random Forest`, `SMOTE + Random Forest`
- 목적: 기존 수업 프로젝트의 성능과 한계를 재현한다.
- 한계: row 단위 random split은 같은 cycle의 유사한 시점이 train/test에 동시에 들어갈 수 있어 성능이 낙관적으로 보일 수 있다.

### 2. Cycle-run-aware split 적용

`cycle_run`을 학습 표본으로 바로 쓰기보다, 우선 평가 분할 기준으로 사용한다.

- 입력 구조는 row-level baseline과 동일하게 유지한다.
- train/test 분할 시 같은 `cycle_run`이 양쪽에 섞이지 않도록 한다.
- 목적: 기존 row random split 결과가 실제 일반화 성능을 과대평가했는지 확인한다.
- 분석 지표: `PR-AUC`, positive-class recall, macro F1, confusion matrix

### 3. Cycle-run 내부 고정 길이 window 구성

시계열 맥락을 직접 반영하는 1차 확장이다.

- window 크기: 5, 10, 20 step
- 경계 조건: `cycle_run` 경계를 넘지 않는다.
- feature: mean, std, min, max, range, first-last delta, simple slope
- label: 기본적으로 window 내부에 positive target이 하나라도 있으면 positive로 둔다.
- 목적: 현재 시점 하나가 아니라 직전 구간의 센서 변화가 이상탐지 성능을 높이는지 확인한다.

이 방식은 딥러닝을 쓰기 전에 수행해야 한다. window feature만으로 성능이 충분히 개선되면, 딥러닝 비교는 성능 향상보다 표현 방식의 차이를 해석하는 실험으로 제한한다.

다만 이 정의는 고장 상태가 이미 포함된 구간을 positive로 두므로, 고장 발생 전 예측으로 해석하지 않는다. `ProtectiveStop`이나 `GripLost`가 한 번 발생한 뒤 `True`가 유지되거나 기록이 끊기는 데이터 특성을 고려하면, 구간 단위 이상탐지와 발생 전 예측은 별도 실험으로 분리해야 한다.

### 3-1. First positive 이전 window 구성

고장 발생 전 예측 가능성을 확인하기 위한 보수적 확장이다.

- 각 `cycle_run`에서 target이 처음 positive가 되는 first positive step을 찾는다.
- first positive step 이후 또는 그 시점을 포함하는 window는 모두 제외한다.
- window가 first positive step 이전에 끝나고, 종료 시점이 prediction horizon 안에 있으면 positive로 둔다.
- prediction horizon은 3, 5, 10 step을 우선 비교한다.
- 이 실험은 이미 발생한 고장 상태를 분류하지 않고, 발생 직전 구간을 구분할 수 있는지 확인한다.
- positive window 수가 적어지므로 단일 split 결과는 반복 split 또는 cross-validation으로 재확인해야 한다.

### 4. Rolling, lag, delta feature 비교

고정 window를 별도 표본으로 만들지 않고 row-level 입력에 시간 특징을 추가하는 방식이다.

- lag feature: 직전 1, 3, 5 step의 센서값
- delta feature: 현재값과 이전값의 차이
- rolling feature: 이동평균, 이동표준편차, 이동범위
- 목적: 기본 머신러닝 모델 안에서 시계열 정보를 어느 정도까지 반영할 수 있는지 확인한다.

### 5. Sequence model은 후속 비교로 제한

`LSTM`, `GRU`, `1D-CNN`은 window feature baseline 이후에 비교한다.

- 입력: cycle 내부 고정 길이 sensor sequence
- 목적: 직접적인 sequence modeling이 window 기반 통계 feature보다 유리한지 확인한다.
- 주의: 데이터 수와 불균형이 크므로 딥러닝 모델이 반드시 더 좋은 결과를 내야 한다고 전제하지 않는다.

## 분석 질문

1. row random split과 cycle-aware split 사이에 성능 차이가 있는가?
2. cycle 내부 window feature가 positive recall 또는 `PR-AUC`를 개선하는가?
3. `System_Failure`를 하나로 묶었을 때와 `ProtectiveStop`, `GripLost`를 분리했을 때 결과가 어떻게 달라지는가?
4. window 크기 5, 10, 20 step 중 어떤 범위가 가장 안정적인가?
5. `ProtectiveStop`과 `GripLost`의 cycle 내 발생 위치 차이가 모델 성능 차이로 이어지는가?
6. 25-cycle acquisition block 하나를 통째로 제외해도 구간 단위 이상탐지 성능이 유지되는가?

## 우선 실험 순서

1. `row baseline` 결과를 기준점으로 고정한다.
2. 같은 feature에서 `random split`과 `cycle_run_group split`을 비교한다.
3. `cycle_run` 경계를 넘지 않는 5, 10, 20 step window feature baseline을 만든다.
4. target을 `System_Failure`, `ProtectiveStop`, `GripLost`로 나누어 같은 실험을 반복한다.
5. window feature 결과가 정리된 뒤, 필요한 경우 `LSTM`, `GRU`, `1D-CNN`을 후속 비교로 추가한다.

## 해석상 제한

- 공개 CSV에 `workload`, `movement speed`, `gripping force`가 직접 컬럼으로 포함되어 있지 않으므로, 이를 모델 입력 feature처럼 설명하지 않는다.
- 25-cycle block은 논문의 반복 횟수를 이용한 acquisition proxy일 뿐 실제 공정조건 조합의 정답 라벨이 아니다.
- 일부 timestamp 간격이 길게 벌어지는 구간이 있으므로 균일한 고속 센서 시계열이라고 쓰지 않는다.
- 연구의 핵심은 복잡한 모델 자체가 아니라, 기존 row 단위 분류가 놓친 시계열 맥락을 어떻게 반영하고 평가할지에 둔다.

## 2026-08-10 재검증 반영

- raw `cycle` 경계를 사용했던 결과에서 window size별 24개, 54개, 104개의 경계-crossing window가 확인되어 `cycle_run` 기준으로 수정했다.
- pre-failure 30회 반복 결과에서 `GripLost`가 상대적으로 가장 안정적이라는 결론은 유지됐지만 절대 성능은 낮았다.
- 9개 acquisition block을 하나씩 제외한 구간 단위 이상탐지에서는 세 target 모두 모든 block에서 positive를 한 번 이상 탐지했다.
- 이 결과는 구간 단위 이상탐지의 block 일반화 근거이며 pre-failure 예측이나 실제 공정조건 식별 근거가 아니다.
