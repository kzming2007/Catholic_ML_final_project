# UR3 CobotOps 로봇팔 센서 데이터의 이상탐지를 위한 시계열 맥락 반영과 딥러닝 모델 비교 연구

> 문서 상태: 연구보고서 초안 v0.1  
> 기준일: 2026-08-27  
> 연구자 정보와 제출 형식은 최종 편집 단계에서 반영한다.

## 초록

산업용 로봇의 센서 데이터는 시간적으로 인접한 관측값이 강하게 연관되므로, 개별 시점을 무작위로 나눈 분류 성능만으로 새로운 운전 구간에 대한 일반화를 판단하기 어렵다. 본 연구는 공개 UR3 CobotOps 데이터셋을 대상으로 cycle과 수집 구간의 경계를 보존한 이상탐지 평가 절차를 구성하고, 짧은 시계열을 통계 특징으로 정제한 Random Forest와 raw sequence 기반 딥러닝 모델을 비교하였다. 비연속적으로 재등장하는 cycle ID를 `cycle_run`으로 분리한 뒤, 경계를 넘지 않는 10-step window와 9개 acquisition block held-out 평가를 사용하였다. 동일한 19개 센서에서 Random Forest는 133개 통계 특징을 사용하고, 1D CNN과 LSTM Autoencoder는 10×19 raw sequence를 입력받았다. Random Forest의 event cycle recall은 `System_Failure` 0.9770, `ProtectiveStop` 0.9516, `GripLost` 0.9231이었다. 완전 정상 cycle 오경보율은 각각 0.0435, 0.0348, 0.0174였다. 1D CNN은 유사한 event recall을 보였지만 완전 정상 오경보와 교차 고장 경보가 증가했다. 정상-only LSTM Autoencoder는 q95 기준 event recall이 0~0.0645로 낮았다. 결과는 복잡한 모델이 자동으로 우수하지 않으며, 현재 데이터 규모에서는 cycle 경계를 보존한 통계 특징과 비선형 분류기의 조합이 더 안정적임을 보여준다. 다만 본 결과는 동일 데이터셋 내부의 block-held-out 검증이며 외부 로봇, 실제 조기경고, 공정조건별 일반화를 입증하지 않는다.

**주요어:** UR3 CobotOps, 다변량 시계열, 이상탐지, cycle-aware split, Random Forest, 1D CNN, LSTM Autoencoder

## 1. 서론

협동로봇은 작업 중 관절 전류, 온도, 속도, 그리퍼 전류와 같은 다변량 센서값을 지속해서 생성한다. 이러한 정보는 보호 정지나 파지 실패를 탐지하는 데 활용할 수 있지만, 시계열의 인접 행을 무작위로 train과 test에 나누면 같은 cycle의 유사한 상태가 양쪽에 포함될 수 있다. 이 경우 높은 분류 성능이 새로운 cycle이나 이후 수집 구간에 대한 일반화를 의미하지 않을 수 있다.

UR3 CobotOps 데이터셋의 소개 연구는 센서 패턴을 이용해 로봇 고장을 분류했으며, 일반적인 stratified cross-validation이 시간 순서를 훼손할 수 있음을 논문에서 직접 지적했다. 이후 연구에서는 feature selection, BiRNN, Transformer와 RUL 예측으로 방법이 확장되었지만, cycle 경계를 독립 표본 단위로 보존한 비교는 충분히 다뤄지지 않았다. 본 연구는 새로운 architecture를 제안하기보다, 공개 데이터에서 시간 경계와 평가 단위를 명확히 설정하고 같은 입력 정보에서 특징 기반 모델과 sequence model을 비교하는 데 목적이 있다.

### 1.1 연구 질문

본 연구는 다음 세 질문을 다룬다.

1. 개별 row 대신 cycle 내부 시계열 window를 사용하면 불균형 고장 분류의 Macro F1과 PR-AUC가 개선되는가?
2. 동일한 10-step×19-sensor 입력에서 통계 특징 기반 Random Forest와 raw sequence 기반 1D CNN·LSTM Autoencoder는 event 탐지와 경보 부담 측면에서 어떤 차이를 보이는가?
3. 관절 전류, 속도, 온도, `Tool_current`의 예측 기여와 모델의 오경보를 어느 범위까지 해석할 수 있는가?

### 1.2 연구 범위

주 태스크는 **이상이 포함된 시계열 구간의 supervised 탐지**다. Window 안에 이미 positive 상태가 포함될 수 있으므로 이를 고장 발생 전 예측으로 표현하지 않는다. First positive 이전 데이터만 사용하는 pre-failure 실험은 별도 부가 분석으로 둔다. 공개 파일에 없는 workload, movement speed, gripping force의 cycle별 정답을 복원하는 작업과 실제 로봇 제어 시스템 구축은 연구 범위에 포함하지 않는다.

## 2. 관련 연구

Tyrovolas et al.은 UR3 CobotOps를 구축하고 `ProtectiveStop OR GripLost`를 `System_Failure`로 정의하여 IF-FCM과 기존 FCM을 비교하였다. 이 연구는 설명 가능성을 중심으로 했으며 SMOTE-ENN과 stratified 10-fold cross-validation을 사용했다. 저자들은 이 방식이 시계열 순서를 훼손할 수 있다고 명시했다.

Yaşar Çıklaçandır et al.은 동일 데이터셋에서 Logistic Regression, Decision Tree, Random Forest, SVM, KNN과 RFE·Chi-Square feature selection을 비교했다. 높은 Random Forest 정확도를 보고했지만 cycle 또는 수집 block을 통째로 제외하는 분할은 제시하지 않았다. Taş and Bal은 Protective Stop 분류에 BiRNN과 hyperparameter 최적화를 적용했으나 row 기준 random 80:20 split을 사용했다. Shen et al.은 chronological split에서 Transformer와 ODE 기반 fault classification·RUL 예측을 수행했다. 이 연구는 현재 연구와 가까운 후속 방향이지만, positive 상태가 지속되는 공개 label을 RUL로 변환하는 절차와 cycle 전체 표준화의 인과적 사용 가능성은 별도 검증이 필요하다.

Pinto Neto는 WAY-EEG-GAL 데이터에서 고전적 spectral decoder와 CNN, Transformer, GNN을 동일 LOSO 조건으로 비교했다. 대상 데이터와 태스크는 다르지만, 작은 다변량 시계열에서 모델 복잡도를 우위로 전제하지 않고 입력 표현·modality·bandwidth를 통제한 비교 설계는 본 연구의 방법론적 참고가 된다.

기존 연구와 비교할 때 본 연구의 차이는 다음과 같다. 첫째, 재등장하는 cycle ID를 실제 연속 구간인 `cycle_run`으로 분리한다. 둘째, window가 cycle 경계를 넘지 않도록 제한한다. 셋째, 9개 acquisition block을 한 번씩 통째로 제외한다. 넷째, 이상 구간 탐지와 pre-failure를 분리한다. 다섯째, target 미발생 경보를 완전 정상 오경보와 다른 고장에 대한 교차 경보로 나눈다.

## 3. 연구 방법

### 3.1 데이터와 타깃

UCI Machine Learning Repository의 UR3 CobotOps 데이터셋을 사용하였다. 공개 CSV는 7,409행×24열이며, 모델 입력에는 관절 J0~J5의 전류·온도·속도와 `Tool_current`를 포함한 원본 센서 19개를 사용하였다. `System_Failure`는 `ProtectiveStop OR GripLost`로 정의하였다. 세 target은 각각 독립적인 binary 탐지 문제로 평가하였다.

원본 `cycle` ID 중 일부는 시간상 떨어진 두 구간에 다시 등장한다. 이를 그대로 group으로 사용하면 실제로 이어지지 않은 시행이 합쳐질 수 있으므로, cycle 값이 바뀌는 연속 구간마다 `cycle_run`을 새로 부여하였다. 최종 공통 비교에는 202개 `cycle_run`이 포함되었다. Event cycle 수는 `System_Failure` 87개, `ProtectiveStop` 62개, `GripLost` 39개다.

### 3.2 시계열 window와 label

각 `cycle_run` 안에서 10-step sliding window를 구성하였다. 서로 겹치는 window는 총 4,035개지만 독립 표본 수로 간주하지 않는다. Window label은 해당 구간 target의 최댓값으로 정의하였다. 따라서 positive window는 이상 발생 시점을 포함할 수 있다. 모델 평가는 window 수준 지표와 202개 cycle 수준 지표를 구분한다.

Row baseline과 초기 window 비교에서는 5·10·20-step을 탐색하였다. 이후 모델 간 공정한 비교에서는 결과 확인 전에 10-step을 공통 입력으로 고정하였다. 10-step은 약 10초라고 단정하지 않고 10개 저장 step으로 해석한다. RTDE 인터페이스의 125 Hz와 공개 CSV의 실제 저장 간격은 같지 않기 때문이다.

### 3.3 분할 전략과 누수 통제

논문의 25회 반복 설명을 바탕으로 만든 acquisition block 중 충분한 cycle이 남은 1, 2, 3, 4, 5, 7, 8, 9, 10번 block을 사용하였다. 각 block을 한 번씩 outer test로 두고 나머지 8개 block으로 학습하였다. Test block은 scaling, SMOTE, epoch 선택, architecture 선택, threshold calibration에 사용하지 않았다.

이 block은 실제 공정조건 정답이 아니라 수집 구간 proxy다. 공개 파일에는 workload, movement speed, gripping force의 cycle별 대응표가 없으며, block은 온도와 session drift의 영향을 함께 포함할 수 있다. 따라서 평가 결과는 동일 데이터셋 안에서 수집 구간을 달리했을 때의 내부 일반화로 제한한다.

### 3.4 비교 모델

#### Random Forest

19개 센서마다 mean, standard deviation, minimum, maximum, range, first-last delta, slope를 계산하여 133개 특징을 만들었다. 학습 block에서만 SMOTE를 적용한 뒤 300-tree Random Forest를 학습하였다. `random_state=42`, decision threshold 0.50을 사용하였다.

#### 1D CNN

10×19 raw sequence를 입력으로 사용하였다. 구조는 Conv1D 19→32, Conv1D 32→64, global average pooling, dropout 0.2, linear output으로 구성하였다. Class-weighted binary cross-entropy, Adam, learning rate `1e-3`, weight decay `1e-4`를 사용하였다. Outer train 중 1개 block으로 epoch를 선택한 뒤 8개 train block 전체에서 다시 학습하였다.

#### LSTM Autoencoder

단층 LSTM encoder와 decoder에 각각 hidden size 32를 사용하고 19개 센서를 복원하도록 구성하였다. `System_Failure=0`인 완전 정상 cycle만 학습하였다. Reconstruction MSE를 anomaly score로 사용하고, 별도 calibration block의 정상 cycle score 95th percentile을 primary threshold로 고정하였다. q90과 q97.5는 민감도 분석으로만 사용하였다.

1D CNN과 LSTM Autoencoder는 seed 42, 43, 44로 반복하였다. Cycle 경보는 세 seed 중 두 개 이상이 경보한 2/3 consensus로 정의하였다. Random Forest는 seed 42의 고정 결과를 사용하였다.

#### 기본 분류 모델 통제 비교

동일한 133개 통계 특징에 Logistic Regression과 RBF SVM을 적용하였다. 두 모델에는 train fold 내부 `StandardScaler`와 SMOTE를 사용하였다. Logistic Regression은 L2, `C=1`, `solver=saga`를, RBF SVM은 `C=1`, `gamma=scale`을 사용하였다. 결과를 확인한 뒤 Grid Search나 threshold tuning을 추가하지 않았다.

### 3.5 센서 그룹 ablation

Random Forest 조건을 유지하면서 전류·속도·온도·`Tool_current` 그룹을 제거하거나 단독으로 사용하였다. 비교 항목과 해석 규칙은 결과 확인 전에 고정하였다. 센서 그룹 변경은 입력 차원과 SMOTE 공간도 함께 바꾸므로, 성능 차이를 센서의 물리적 인과효과로 해석하지 않는다.

### 3.6 평가 지표

Window 수준에서는 Accuracy, Macro F1, positive precision·recall·F1, ROC-AUC, PR-AUC를 계산하였다. Cycle 수준의 주 지표는 다음과 같다.

- **Event cycle recall:** 실제 positive window가 있는 cycle 중 positive window를 하나 이상 탐지한 비율
- **Target-negative alert rate:** 해당 target이 없는 모든 cycle에서 경보가 발생한 비율
- **True-normal false-alarm rate:** 어떤 고장도 없는 cycle에서 경보가 발생한 비율
- **Cross-fault alert rate:** 해당 target은 없지만 다른 고장만 발생한 cycle에서 경보가 발생한 비율

Cycle 비율에는 Wilson 95% confidence interval을 계산하고, block별 event recall 최솟값과 false-alarm rate 최댓값을 함께 확인하였다. 다만 9개 block은 서로 다른 로봇이나 독립 수집 반복이 아니라 동일 기록을 나눈 평가 단위다. 따라서 현재 초안에서는 block을 독립 모집단 표본으로 가정한 유의성 검정을 주 근거로 사용하지 않는다. 모델 간 paired test가 필요할 경우에는 제한된 사후 민감도 분석으로 명시하고 다중비교 보정을 적용해야 한다.

## 4. 연구 결과

### 4.1 Row baseline과 시계열 window 비교

| Target | Row Macro F1 | Best window | Window Macro F1 | Row PR-AUC | Window PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `System_Failure` | 0.7901 | 5-step | 0.8166 | 0.6430 | 0.8613 |
| `ProtectiveStop` | 0.8428 | 5-step | 0.8642 | 0.7409 | 0.9039 |
| `GripLost` | 0.7623 | 10-step | 0.8412 | 0.5604 | 0.8493 |

세 target 모두 window 특징에서 Macro F1과 PR-AUC가 개선되었다. 이는 한 시점의 센서값보다 짧은 구간의 수준, 변동성, 변화량이 고장 상태를 구분하는 데 유용할 수 있음을 보여준다. 다만 target별 최적 window 크기가 달랐으므로 하나의 길이가 모든 고장에 최적이라고 결론 내리지 않는다.

### 4.2 동일 센서 기반 주 모델 비교

| Target | 모델 | Event recall | 완전 정상 오경보율 | 교차 고장 경보율 |
| --- | --- | ---: | ---: | ---: |
| `System_Failure` | Random Forest | 0.9770 | 0.0435 | 해당 없음 |
|  | 1D CNN | 0.9770 | 0.1913 | 해당 없음 |
|  | LSTM Autoencoder q95 | 0.0460 | 0.1304 | 해당 없음 |
| `ProtectiveStop` | Random Forest | 0.9516 | 0.0348 | 0.0000 |
|  | 1D CNN | 0.9355 | 0.2087 | 0.1200 |
|  | LSTM Autoencoder q95 | 0.0645 | 0.1304 | 0.0000 |
| `GripLost` | Random Forest | 0.9231 | 0.0174 | 0.0833 |
|  | 1D CNN | 0.9231 | 0.1652 | 0.4375 |
|  | LSTM Autoencoder q95 | 0.0000 | 0.1304 | 0.1042 |

Random Forest는 세 target에서 높은 event recall과 낮은 완전 정상 오경보를 함께 유지하였다. 1D CNN은 event recall이 비슷했지만 경보 부담이 증가했다. 특히 `GripLost`에서는 다른 고장만 발생한 cycle의 43.75%에 경보하여 고장 유형 간 구분이 충분하지 않았다. LSTM Autoencoder는 정상 재구성 오차가 고장 상태를 일관되게 분리할 것이라는 가정을 지지하지 못했다.

Window 수준에서도 Random Forest의 PR-AUC는 `System_Failure` 0.8137, `ProtectiveStop` 0.8557, `GripLost` 0.7267로 1D CNN의 0.6879, 0.7659, 0.5888보다 높았다. Autoencoder q95의 PR-AUC는 각각 0.2228, 0.1629, 0.0834였다.

### 4.3 기본 분류 모델 통제 비교

| Target | 모델 | Event recall | 완전 정상 오경보율 | 교차 고장 경보율 |
| --- | --- | ---: | ---: | ---: |
| `System_Failure` | Random Forest | 0.9770 | 0.0435 | 해당 없음 |
|  | Logistic Regression | 0.9885 | 0.5304 | 해당 없음 |
|  | RBF SVM | 0.9540 | 0.0957 | 해당 없음 |
| `ProtectiveStop` | Random Forest | 0.9516 | 0.0348 | 0.0000 |
|  | Logistic Regression | 0.9677 | 0.2696 | 0.2800 |
|  | RBF SVM | 0.9032 | 0.0174 | 0.0400 |
| `GripLost` | Random Forest | 0.9231 | 0.0174 | 0.0833 |
|  | Logistic Regression | 1.0000 | 0.4522 | 0.5833 |
|  | RBF SVM | 0.9487 | 0.0957 | 0.3750 |

Logistic Regression은 높은 event recall을 보였지만 완전 정상 cycle과 다른 고장 cycle에도 자주 경보했다. RBF SVM은 경보를 줄였으나 target별 trade-off가 남았다. 통계 특징에 선형·kernel 모델도 활용할 수 있는 신호가 존재하지만, 현재 조건에서는 Random Forest가 탐지율과 경보 부담을 가장 안정적으로 균형화하였다.

### 4.4 센서 그룹 ablation

관절 전류 계열을 제거하면 `System_Failure` event recall은 0.9770에서 0.9310으로, `GripLost`는 0.9231에서 0.4103으로 감소하였다. 반면 `ProtectiveStop` recall은 0.9677로 높아졌지만 완전 정상 오경보율 0.0609와 교차 고장 경보율 0.1600이 함께 증가하였다. `Tool_current`만 제거했을 때 `System_Failure`와 `ProtectiveStop`은 소폭 개선되었고, `GripLost`의 교차 고장 경보율은 0.0833에서 0.1042로 증가하였다.

이 결과는 관절 전류 계열이 특히 `GripLost` 탐지에 기여한다는 해석을 지지한다. 그러나 모든 전류 센서가 항상 필요하거나 고장의 원인이라고 결론 내릴 수는 없다. 온도 단독 모델의 탐지력은 낮았지만 온도 제거 후 일부 경보가 증가했으며, 이는 온도가 고장 자체보다 수집 구간의 상태를 보정하는 변수로 작동했을 가능성을 남긴다.

### 4.5 Pre-failure 부가 분석

First positive 이전 window만 사용한 30회 반복 split에서 positive F1 평균은 `System_Failure` 0.0920, `ProtectiveStop` 0.0867, `GripLost` 0.3058이었다. 적어도 하나의 positive를 탐지한 split 비율은 `GripLost`에서 0.9667이었지만 평균 recall은 0.2248이었다. 따라서 `GripLost`에 약한 사전 신호가 있을 가능성은 남지만 실용적인 조기경고 성능을 확보했다고 볼 수 없다.

### 4.6 공정조건 역추적 결과

논문에는 workload 1/2/3 kg, movement speed 60/80/100%, gripping force 80/100/120 N과 scenario당 25회 반복이 기술되어 있다. 그러나 공개 CSV에는 cycle별 조건표가 없고 가능한 27개 조합의 사용 여부도 제시되지 않았다. 센서 proxy 기반 clustering은 세 수준과 일치하지 않았으며, 25-cycle 후보 block은 공정 센서보다 온도만으로 더 잘 구분되었다. 따라서 공정조건 역추적은 본 연구의 성과가 아니라 데이터 provenance의 한계와 session drift 가능성을 확인한 감사 결과로 해석한다.

## 5. 논의

### 5.1 시계열 맥락의 효과

Row baseline보다 window 특징의 PR-AUC가 일관되게 증가한 결과는 짧은 시계열 맥락이 유용하다는 연구 질문에 긍정적인 답을 제공한다. 다만 이 개선은 단순히 window를 도입한 효과만은 아니다. Window는 7개 통계량을 통해 원본 센서값을 133개 특징으로 확장하므로, 시간 맥락과 feature engineering의 효과가 결합되어 있다.

### 5.2 딥러닝 비교의 의미

1D CNN과 LSTM Autoencoder가 Random Forest를 넘지 못한 결과는 딥러닝이 부적절하다는 일반 결론을 뜻하지 않는다. 현재 데이터는 단일 로봇 기록, 202개 cycle, 10-step sequence로 제한된다. 이 조건에서는 통계 특징이 작은 데이터의 변동을 안정적으로 요약했고, Random Forest가 비선형 상호작용을 효과적으로 처리한 것으로 해석할 수 있다. 반면 1D CNN은 고장 신호와 다른 고장 유형의 공통 패턴을 함께 포착하면서 경보가 증가했을 가능성이 있다.

LSTM Autoencoder는 정상 분포만으로 모든 고장을 정의할 수 있다는 가정에 의존한다. Calibration에 사용할 수 있는 정상 cycle이 fold별 13~36개로 적었고 block drift도 존재했다. q90, q95, q97.5에서 결론이 바뀌지 않았으므로 threshold를 사후 조정해 주 결과를 교체하지 않았다.

### 5.3 학술적 기여

본 연구의 기여는 새로운 모델 architecture보다 평가 설계에 있다. 관련 연구의 row random split이나 일반 K-fold 결과를 그대로 재사용하지 않고, 실제 연속 시행과 수집 구간을 보존하였다. 또한 event 탐지와 pre-failure를 분리하고, 기존의 target-negative 경보율을 완전 정상 오경보와 교차 고장 경보로 세분화하였다. 이를 통해 높은 recall 뒤에 숨은 운영상 경보 부담과 고장 유형 혼동을 함께 제시할 수 있었다.

## 6. 연구의 한계

1. 하나의 공개 데이터셋과 한 대의 UR3에서 수집된 기록만 사용하였다.
2. 9개 acquisition block은 독립적인 로봇·기관·반복 실험이 아니므로 외부 일반화를 검증하지 못한다.
3. 4,035개 window는 서로 겹치며 독립 표본 수가 아니다. 주 해석은 202개 cycle을 기준으로 한다.
4. 공개 파일에는 cycle별 workload, movement speed, gripping force 정답표가 없다.
5. `ProtectiveStop`과 `GripLost`가 동시에 나타나는 행이 있어 단일 3-class 문제로 바꾸려면 별도의 처리 규칙이 필요하다.
6. Random Forest는 고정 seed 1회, 딥러닝은 3개 seed consensus이므로 초기화 반복 조건이 완전히 같지 않다.
7. LSTM Autoencoder threshold calibration의 정상 cycle 수가 적다.
8. Sensor ablation은 feature 차원과 SMOTE 공간을 함께 바꾸므로 인과적 센서 중요도를 증명하지 않는다.
9. Pre-failure 결과는 약한 탐색적 신호이며 실제 조기경고 성능이 아니다.
10. 실시간 추론 지연, 연속 경보 정책, 새로운 고장 유형 대응을 검증하지 않았다.
11. 현재 분석은 내부 재집계 검증을 마쳤지만, 깨끗한 별도 작업공간에서의 전체 CPU·GPU 재실행은 아직 완료하지 않았다.

## 7. 결론

본 연구는 UR3 CobotOps 센서 데이터에서 cycle과 수집 구간 경계를 보존한 시계열 이상탐지 평가를 수행하였다. 짧은 window의 통계 특징은 row baseline보다 Macro F1과 PR-AUC를 개선하였다. 동일한 10-step×19-sensor 입력 비교에서 Random Forest는 1D CNN과 LSTM Autoencoder보다 높은 event recall과 낮은 경보 부담을 안정적으로 유지하였다. 센서 그룹 분석에서는 관절 전류의 예측 기여가 확인되었지만, 모든 전류 센서의 일관된 필요성이나 물리적 인과관계는 지지되지 않았다.

따라서 현재 데이터 규모에서는 복잡한 딥러닝 모델을 추가하는 것보다, 독립 표본의 경계를 보존하고 경보의 의미를 구분하는 평가 설계가 더 중요한 것으로 판단된다. 후속 연구에서는 별도 로봇 또는 신규 수집 세션을 이용한 외부 검증, 공정조건 대응표 확보, pre-failure label 정의의 개선이 필요하다.

## 8. 재현성 및 검증 계획

- 데이터 파일: `dataset/ur3_cobotops.csv`
- SHA-256: `C789CDA10ACB354A7C1689F617D94A5F39A93FD8CB6C004AD16D36CEA55A74A3`
- CPU 환경: Python 3.12.3, NumPy 2.4.3, pandas 3.0.3, scikit-learn 1.8.0, imbalanced-learn 0.14.1
- 딥러닝 환경: PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 3060 Ti
- 공통 seed: Random Forest 42, 딥러닝 42·43·44
- 핵심 결과 재집계: `13_final_evaluation_tables.py`, `16_fault_context_alert_analysis.py`
- 보고서 수치 대조: `17_report_evidence_validation.py`

현재 단계에서는 저장된 prediction과 결과 CSV를 이용한 내부 일관성 검증을 수행한다. 전체 재현 검증에서는 별도 clean worktree에서 CPU 전처리·분류 스크립트를 순서대로 실행하고, 외부 PyTorch 환경에서 seed별 딥러닝 학습을 다시 수행해야 한다. 결정적 CPU 결과는 exact match를, GPU 결과는 cycle prediction 일치와 headline metric 허용오차를 기준으로 비교한다. 세부 절차는 `2026-08-27_reproducibility_review_plan.md`에 기록한다.

## 참고문헌

1. Tyrovolas, M., Stylios, C., Aliev, K., & Antonelli, D. (2024). Leveraging Information Flow-Based Fuzzy Cognitive Maps for Interpretable Fault Diagnosis in Industrial Robotics. DOI `10.1007/978-3-031-63851-0_6`.
2. Tyrovolas, M., Aliev, K., Antonelli, D., & Stylios, C. (2024). UR3 CobotOps. UCI Machine Learning Repository. DOI `10.24432/C5J891`.
3. Yaşar Çıklaçandır, F. G., Mumcu, S. A., Çam, B., & Ceran, I. (2025). Impact of Feature Selection on the Performance of Classification Algorithms in Predicting Industrial Robot Failures. DOI `10.21205/deufmd.2025278107`.
4. Taş, G., & Bal, C. (2025). An Egret Swarm Optimization Based BiRNN Method Approach to Determine the Protective Stopping Time of UR3 Robot Arm. DOI `10.16984/saufenbilder.1755797`.
5. Shen, Y., et al. (2026). Physics-Informed Transformer with ODE-Guided Joint Modeling for Fault Classification and RUL Prediction in Collaborative Robots. DOI `10.36001/phmap.2025.v5i1.4625`.
6. Pinto Neto, O. (2026). Architecture-data matching for EEG-EMG decoding: compact deep models match classical spectral decoders on the WAY-EEG-GAL grasp-and-lift dataset. DOI `10.3389/fnins.2026.1874302`.
7. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. DOI `10.1162/neco.1997.9.8.1735`.
8. Malhotra, P., et al. (2016). LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection. arXiv `1607.00148`.
9. Fawaz, H. I., et al. (2019). Deep Learning for Time Series Classification: A Review. DOI `10.1007/s10618-019-00619-1`.
