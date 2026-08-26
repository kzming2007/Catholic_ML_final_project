# UR3 CobotOps 관련 연구 및 방법론 참고 검토

## 검토 목적

현재 연구와 직접 겹치는 UR3 CobotOps 데이터셋 활용 연구와, 다른 데이터지만 비교 설계에 참고할 수 있는 논문을 구분한다. 문헌의 성능 수치를 그대로 기준점으로 사용하지 않고 분할 단위, 타깃 정의, 시간 맥락, 누수 통제 수준을 먼저 비교한다.

## 직접 관련 연구

### 1. 데이터셋 소개 및 IF-FCM 연구

- Tyrovolas et al. (2024), *Leveraging Information Flow-Based Fuzzy Cognitive Maps for Interpretable Fault Diagnosis in Industrial Robotics*, DOI `10.1007/978-3-031-63851-0_6`.
- `ProtectiveStop OR GripLost`를 `System Failure`로 합친 row-level binary classification이다.
- `Timestamp`와 `Cycle`을 제거하고 SMOTE-ENN과 stratified 10-fold cross-validation을 사용했다.
- 논문 자체가 time-series에서 SMOTE-ENN과 일반 K-fold가 시간 의존성을 훼손할 수 있음을 인정하며, 시간 순서가 아니라 센서 패턴 분류에 초점을 둔다.
- 현재 연구는 이 한계를 직접 이어받아 `cycle_run` 경계와 acquisition block held-out 평가를 도입했다.

### 2. Feature selection 기반 기본 분류 연구

- Yaşar Çıklaçandır et al. (2025), *Impact of Feature Selection on the Performance of Classification Algorithms in Predicting Industrial Robot Failures*, DOI `10.21205/deufmd.2025278107`.
- `ProtectiveStop`과 `GripLost`를 각각 분류하고 Logistic Regression, Decision Tree, Random Forest, SVM, KNN을 비교했다.
- StandardScaler, SMOTE, RFE와 Chi-Square top-10 feature selection을 사용했으며 데이터는 70:30으로 분할했다.
- `Cycle`을 입력 feature에 포함했고, cycle 또는 수집 block을 통째로 제외하는 분할은 보고하지 않았다. 전처리와 feature selection이 train fold 내부에서만 수행됐는지도 본문만으로 확정하기 어렵다.
- 약 0.99의 Random Forest 정확도는 현재 block-held-out cycle 평가와 직접 비교할 수 없다. 대신 기본 모델과 센서 선택을 비교해야 한다는 문제의식은 현재 `14`·`15`와 직접 겹친다.

### 3. BiRNN 기반 protective-stop 분류 연구

- Taş and Bal (2025), *An Egret Swarm Optimization Based BiRNN Method Approach to Determine the Protective Stopping Time of UR3 Robot Arm*, DOI `10.16984/saufenbilder.1755797`.
- Protective stop을 대상으로 BiRNN과 여러 딥러닝 모델을 비교하고 Egret Swarm Optimization으로 learning rate, unit 수, batch size 등을 탐색했다.
- 데이터는 row 기준 random 80:20으로 분할됐으며 cycle 경계 보존은 보고되지 않았다. 따라서 인접 시점과 같은 cycle이 train/test에 함께 들어갈 수 있다.
- 딥러닝을 사용했다는 점은 직접 관련되지만, 현재 연구의 9-block held-out 결과와 ROC-AUC 0.9461을 같은 일반화 성능으로 비교하지 않는다.

### 4. Transformer와 RUL의 후속 연구

- Shen et al. (2026), *Physics-Informed Transformer with ODE-Guided Joint Modeling for Fault Classification and RUL Prediction in Collaborative Robots*, DOI `10.36001/phmap.2025.v5i1.4625`.
- Normal, Grip Loss, Protective Stop의 3-class fault classification과 다음 고장까지의 RUL을 함께 학습한다.
- 70/15/15 chronological split을 사용하고 Transformer, GRU, LSTM, CNN, CNN-LSTM과 비교했다. 보고된 test macro-F1은 0.2731이다.
- 현재 연구와 가장 가까운 후속 방향이지만, 공개 데이터의 positive 상태 지속 구간을 next-failure/RUL label로 어떻게 처리했는지와 전체 cycle 통계로 표준화했는지는 재현 전에 별도 감사가 필요하다.
- 현재 연구의 pre-failure 실험은 first positive 이후를 제외한다는 점에서 더 보수적이지만, 실용적 조기경고 성능을 확보하지는 못했다.

## 방법론 참고 논문

### Architecture-data matching for EEG-EMG decoding

- Pinto Neto (2026), *Architecture-data matching for EEG-EMG decoding: compact deep models match classical spectral decoders on the WAY-EEG-GAL grasp-and-lift dataset*, DOI `10.3389/fnins.2026.1874302`.
- 12명, 3,528 trial에서 물체 무게와 표면 마찰을 각각 3-class로 분류한다.
- 고전적 spectral feature 모델, MLP, compact CNN, Transformer, graph attention model을 동일한 LOSO split에서 비교한다.
- balanced accuracy와 macro-F1, Friedman test, paired Wilcoxon test, BH-FDR 보정을 사용하며 bandwidth, phase, modality, conditioning ablation을 수행한다.
- compact deep model은 최상위 고전 모델을 통계적으로 넘지 못했고, Transformer는 작은 표본 규모에서 열세였다. EMG-only가 EEG-EMG fusion보다 나은 결과도 있어 정보가 적은 채널 추가가 성능을 악화시킬 수 있음을 보여준다.

## 현재 연구와의 공통점 및 차이

| 항목 | EEG-EMG 논문 | 현재 UR3 연구 |
| --- | --- | --- |
| 데이터 | 사람 12명의 trial | 단일 공개 로봇 기록의 202 cycle_run |
| 주 태스크 | 무게·표면 3-class 분류 | 고장 포함 구간의 target별 이상탐지 |
| 독립 단위 | held-out subject | held-out acquisition block |
| 표현 비교 | spectral summary vs raw 1초 sequence | 10-step 통계 특징 vs raw sequence |
| 모델 비교 | 고전 모델, CNN, Transformer, GNN | RF, LR, RBF SVM, 1D CNN, LSTM AE |
| 해석 보조 | modality·bandwidth·phase ablation | sensor group ablation, fault-context alert 분리 |
| 핵심 결론 | 복잡한 모델이 자동으로 우세하지 않음 | RF 통계 특징이 현재 데이터에서 더 안정적임 |

공통된 연구 논리는 `같은 분할과 입력 정보에서 표현 또는 모델만 바꾸고, 복잡한 모델의 우위를 전제하지 않는다`는 것이다. 다만 EEG-EMG 논문의 LOSO는 보지 못한 사람에 대한 일반화이고, 현재 block split은 동일 데이터 수집 내 구간 일반화다. 근거 수준을 같다고 표현하지 않는다.

## 현재 연구의 위치

현재 연구는 새로운 architecture를 제안하지 않는다. 직접 관련 연구가 주로 row random split, 일반 stratified K-fold, 또는 단일 chronological split을 사용한 것과 달리 다음 항목을 명시적으로 다룬다.

1. 재등장하는 원본 cycle ID를 `cycle_run`으로 분리한다.
2. window가 cycle 경계를 넘지 않게 한다.
3. 9개 acquisition block을 하나씩 통째로 제외한다.
4. 이상이 포함된 구간 탐지와 first-positive 이전 예측을 분리한다.
5. 완전 정상 오경보와 다른 고장에 대한 교차 경보를 분리한다.
6. 같은 19개 센서에서 통계 특징 기반 모델과 raw sequence 모델을 비교한다.

따라서 학술적 차별점은 더 복잡한 모델 자체가 아니라, 이 데이터셋에서 시간 경계와 경보 의미를 보존한 비교 설계에 있다.

## 참고문헌

1. Tyrovolas, M., Stylios, C., Aliev, K., & Antonelli, D. (2024). Leveraging Information Flow-Based Fuzzy Cognitive Maps for Interpretable Fault Diagnosis in Industrial Robotics. DOI `10.1007/978-3-031-63851-0_6`.
2. Tyrovolas, M., Aliev, K., Antonelli, D., & Stylios, C. (2024). UR3 CobotOps. UCI Machine Learning Repository. DOI `10.24432/C5J891`.
3. Yaşar Çıklaçandır, F. G., Mumcu, S. A., Çam, B., & Ceran, I. (2025). Impact of Feature Selection on the Performance of Classification Algorithms in Predicting Industrial Robot Failures. DOI `10.21205/deufmd.2025278107`.
4. Taş, G., & Bal, C. (2025). An Egret Swarm Optimization Based BiRNN Method Approach to Determine the Protective Stopping Time of UR3 Robot Arm. DOI `10.16984/saufenbilder.1755797`.
5. Shen, Y., et al. (2026). Physics-Informed Transformer with ODE-Guided Joint Modeling for Fault Classification and RUL Prediction in Collaborative Robots. DOI `10.36001/phmap.2025.v5i1.4625`.
6. Pinto Neto, O. (2026). Architecture-data matching for EEG-EMG decoding: compact deep models match classical spectral decoders on the WAY-EEG-GAL grasp-and-lift dataset. DOI `10.3389/fnins.2026.1874302`.
