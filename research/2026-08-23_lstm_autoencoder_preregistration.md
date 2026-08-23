# 2026-08-23 LSTM Autoencoder 후속 실험 사전 고정 기록

## 문서 목적

이 문서는 `12`번 후속 실험의 결과를 확인하기 전에 연구 질문, 데이터 분할, 입력, 모델, 임계값, 평가 지표와 해석 기준을 고정한다. 로컬 Git 커밋으로 실행 전 상태를 보존한다. 정식 외부 사전등록 저장소를 사용한 것은 아니므로 최종 보고서에서는 `사전 고정 기록`으로 표현한다.

## 연구 질문과 위계

주 연구 결과는 `09`-`11`에서 검증한 **구간 단위 supervised 이상탐지 비교**로 유지한다. 후속 질문은 다음과 같다.

> 정상 로봇 cycle의 센서 sequence만 학습한 LSTM Autoencoder가 같은 10-step 구간에서 Random Forest 및 1D CNN과 비교해 이상 event를 탐지하면서 정상 cycle 오경보를 억제할 수 있는가?

LSTM Autoencoder는 보조·탐색 실험이다. 성능이 좋을 때만 채택하지 않고, 우수·유사·열세 결과를 모두 기록한다. 결과에 따라 연구 질문이나 primary threshold를 변경하지 않는다.

## LSTM 선택 근거

- LSTM은 순환학습에서 장기간 정보 전달이 약해지는 문제를 다루기 위해 제안된 sequence model이다.
- LSTM Encoder-Decoder는 정상 multivariate time series를 재구성하도록 학습하고 reconstruction error를 anomaly score로 사용하는 선행 방법이 있다.
- 본 데이터의 10-step은 긴 sequence가 아니므로 `장기 기억이 반드시 필요하다`고 주장하지 않는다. LSTM은 순서를 보존하는 정상패턴 재구성 방법의 대표 비교군으로 사용한다.
- GRU와 Transformer는 추가하지 않는다. 결과가 좋은 architecture를 탐색하는 것을 막고 학부 연구의 최소 범위를 유지하기 위해서다.

## 데이터와 실질 표본 수

- 데이터: UCI UR3 CobotOps 공개 데이터셋.
- 포함 block: cycle 번호 기반 완전 후보 block `1, 2, 3, 4, 5, 7, 8, 9, 10`.
- 입력 window: `cycle_run` 경계를 넘지 않는 10-step.
- 입력 feature: 파생변수를 제외한 원본 센서 19개.
  - `Current_J0`-`Current_J5`
  - `Temperature_J0`-`Temperature_J5`
  - `Speed_J0`-`Speed_J5`
  - `Tool_current`
- 평가 단위: 202개 `cycle_run`.
  - `System_Failure`: event 87, normal 115.
  - `ProtectiveStop`: event 62, normal 140.
  - `GripLost`: event 39, normal 163.
- 4,035개 window는 서로 겹치므로 독립적인 연구 표본 수로 해석하지 않는다.
- seed 반복은 학습 초기화 민감도 확인용이며 데이터 표본 수에 포함하지 않는다.

## 공통 outer 평가

9개 block을 한 번씩 outer test로 두고 각 `cycle_run`이 정확히 한 번만 test prediction을 갖게 한다. Test block은 scaling, epoch, architecture, threshold 선택에 사용하지 않는다.

### Random Forest

- Outer test 1개 block, train 8개 block.
- 19개 원본 센서의 window mean, std, min, max, range, first-last delta, slope를 입력으로 사용한다.
- 모델: `SMOTE + Random Forest`, 300 trees, `random_state=42`.
- Decision rule: `score > 0.50`.

### 1D CNN

- Outer train 중 시간순 다음 block 1개를 validation으로 두고 나머지 7개로 epoch를 선택한다.
- 선택한 epoch만큼 outer train 8개 block 전체에서 같은 seed로 처음부터 재학습한다.
- 구조: Conv1D 19→32, Conv1D 32→64, global average pooling, dropout 0.2, linear output.
- Loss: train label 비율에서 계산한 class-weighted BCE. `pos_weight = negative windows / positive windows`.
- Decision rule: `score > 0.50`.

### LSTM Autoencoder

- 한 outer test fold에서 다음 순서로 block 역할을 고정한다.
  - Early-stopping validation: test 다음 1개 block.
  - Threshold calibration: 그다음 2개 block.
  - Core train: 남은 5개 block.
- Core train과 validation에서는 `System_Failure=0`인 완전 정상 cycle의 window만 사용한다.
- Core train 5개 block으로 epoch를 선택한 뒤, core train+validation 6개 block의 정상 window에서 선택 epoch만큼 처음부터 재학습한다.
- Threshold calibration 2개 block은 재학습에 사용하지 않는다.
- Calibration에도 `System_Failure=0`인 완전 정상 cycle만 사용한다.
- 구조: 단층 LSTM encoder hidden 32, 단층 LSTM decoder hidden 32, 19-feature linear reconstruction output.
- Loss: train-only z-score 표준화 후 10-step×19-feature 전체 MSE 평균.
- Class weight와 SMOTE는 사용하지 않는다.
- Window anomaly score: 표준화 공간의 window reconstruction MSE.
- Cycle calibration score: 정상 cycle 내부 window anomaly score의 최댓값.
- Primary threshold: calibration cycle score의 empirical 95th percentile, `method="higher"`.
- Sensitivity threshold: 90th, 97.5th percentile. 세 결과를 모두 보고하며 최적값을 사후 선택하지 않는다.
- Decision rule: `score > threshold`.

## 고정 학습 설정

- Seeds: `42, 43, 44`.
- Batch size: 256.
- Optimizer: Adam.
- Learning rate: `1e-3`.
- Maximum epochs: 40.
- Minimum epochs before stopping: 5.
- Early-stopping patience: 5.
- 1D CNN weight decay: `1e-4`.
- LSTM Autoencoder weight decay: 0.
- Scaling parameter는 해당 fold의 train 데이터에서만 계산한다.
- Test 결과를 확인한 뒤 architecture, learning rate, loss, threshold quantile, window 크기를 변경하지 않는다.

## 평가 지표

### Primary

- Event cycle recall: 실제 positive window가 있는 cycle에서 positive window를 하나 이상 탐지한 비율.
- Normal cycle false-alarm rate: 실제 positive window가 없는 cycle에서 경보가 하나 이상 발생한 비율.

### Secondary

- Window Macro F1.
- Window positive precision, recall, F1.
- Window PR-AUC.
- Held-out block별 event recall 최솟값.
- Held-out block별 정상 cycle false-alarm rate 최댓값.
- Seed별 결과와 2/3 seed cycle consensus 결과.
- Cycle-level 비율의 Wilson 95% confidence interval.

## 결과 해석 규칙

- Autoencoder가 더 좋아야 연구가 성공한 것으로 정의하지 않는다.
- 같은 수준의 정상 cycle 오경보에서 event recall이 높으면 정상패턴 재구성의 추가 가능성으로 해석한다.
- Event recall이 비슷하지만 오경보가 높으면 실질적 이점이 없는 것으로 해석한다.
- Event recall과 오경보가 모두 불리하면 현재 데이터에서는 Random Forest를 우선하는 근거로 해석한다.
- 한 모델이 recall과 오경보에서 모두 유리한 경우에만 단순 우위로 표현한다. 지표가 엇갈리면 trade-off로 보고한다.
- Block 최솟값과 최댓값이 불안정하면 전체 평균만으로 일반화 성능을 주장하지 않는다.
- Autoencoder sensitivity threshold 중 가장 좋은 값만 최종 성능처럼 선택하지 않는다.

## 근거와 한계

- LSTM 원 논문: Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. DOI `10.1162/neco.1997.9.8.1735`.
- 정상 sequence 재구성 기반 이상탐지: Malhotra et al. (2016). LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection. arXiv `1607.00148`.
- Optimizer: Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv `1412.6980`.
- 불균형 평가에서 PR 지표 근거: Saito, T., & Rehmsmeier, M. (2015). DOI `10.1371/journal.pone.0118432`.
- 95th percentile threshold는 보편적 최적값이 아니라 정상 cycle 오경보 약 5%를 목표로 정한 primary 운영 기준이다. Calibration 표본이 작고 block drift가 있어 실제 test 오경보 5%를 보장하지 않는다.
- 동일 데이터의 기존 결과를 이미 확인했으므로 이 실험은 독립 외부 검증이 아니다. `사전 고정한 동일 데이터 내부 후속 비교`로 제한한다.
- 강한 외부 일반화 주장은 신규 UR3 실험 또는 이전에 사용하지 않은 독립 데이터가 확보된 뒤에만 가능하다.
