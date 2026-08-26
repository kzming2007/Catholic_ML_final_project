# 기말 프로젝트에서 학술연구로의 확장 정리

## 문서 목적

이 문서는 기존 머신러닝 기말 프로젝트와 학술연구 확장 단계의 역할을 구분하고, 현재까지 실제로 얻은 결과와 한계를 한곳에 정리한다. 기존 보고서의 표현을 그대로 반복하지 않고, 현재 코드와 재실행 결과로 확인된 범위만 연구 결론으로 사용한다.

## 한 문장 결론

**기존 프로젝트는 row 단위 분석에서 Random Forest와 전류 계열 변수의 가능성을 찾았고, 후속 연구는 cycle 경계와 수집 block을 고려한 검증을 통해 짧은 센서 시계열을 통계 특징으로 정제한 Random Forest가 1D CNN과 정상-only LSTM Autoencoder보다 안정적임을 확인했다. 다만 현재 결과는 이상이 포함된 구간의 탐지이며, 독립적인 조기 고장 예측이나 실제 공정조건 식별을 증명한 것은 아니다.**

## 1. 근거의 위계

현재 연구에서는 자료의 역할을 다음과 같이 구분한다.

1. `dataset/`, `notebooks/`, `reports/`: 수업 프로젝트의 원본 자료와 당시 분석 결과
2. `research/00`-`12`: 후속 연구에서 실제로 실행한 재검증 코드
3. `research/outputs/`: 현재 수치와 결론의 직접 근거
4. 연구계획서와 방법론 메모: 실행 전 목표와 분석 기준

원본 보고서의 수치는 당시 프로젝트 성과로 보존한다. 그러나 후속 연구와 분할 방식이나 평가 단위가 다른 수치를 현재 일반화 성능처럼 사용하지 않는다.

## 2. 기존 기말 프로젝트가 수행한 범위

| 태스크 | 질문 | 주요 방법 | 당시 핵심 결과 | 후속 연구에서 확인한 한계 |
| --- | --- | --- | --- | --- |
| Regression | 관절 센서로 `Tool_current`를 예측할 수 있는가? | Linear Regression, Random Forest, KNN | Random Forest `R²=0.5848`; `Tool_current`의 삼봉 분포와 비선형성 확인 | 시계열 일반화보다 row 단위 수치 예측에 초점을 둠. 상태 전환 문제를 직접 검증한 것은 아님 |
| Classification | `System_Failure`를 분류할 수 있는가? | Logistic Regression, SVM, Decision Tree, Random Forest, KNN, SMOTE | Row random split에서 Random Forest Macro F1 약 `0.80`, ROC-AUC 약 `0.94` | 동일 cycle의 인접 row가 train/test에 섞일 수 있고, 통합 라벨이 고장 유형 차이를 가림 |
| Clustering | 센서값으로 운영 상태를 나눌 수 있는가? | K-Means, DBSCAN, Hierarchical Clustering | 전류 7개 기반 K-Means `K=4`, Silhouette `0.3814`; 특정 군집의 Protective Stop 비율 `14.34%` | 군집 명칭은 사후 해석이며 실제 상태 정답이 아님. 고장과 같은 row에서의 연관성은 조기경고 증거가 아님 |

### 기존 프로젝트에서 유지할 수 있는 발견

- 데이터는 정상 약 93%, 고장 약 7%로 불균형하다.
- 선형 모델보다 Random Forest가 당시 row-level 분류에서 더 높은 성능을 보였다.
- 전류 계열 변수는 여러 분석에서 반복적으로 유용한 지표로 나타났다.
- `Tool_current`와 관절 센서 관계는 단순한 선형 관계로 설명하기 어렵다.
- 정적인 row 분석만으로는 시간에 따른 변화와 고장 전후의 순서를 구분할 수 없다.

### 기존 표현에서 그대로 계승하면 안 되는 부분

- Random row split의 높은 점수를 보지 못한 cycle이나 수집 session에 대한 성능으로 일반화하지 않는다.
- Random Forest feature importance를 고장의 물리적 원인이나 인과관계로 해석하지 않는다.
- K-Means 군집을 실제 `정상`, `과부하`, `그리퍼 하중`, `특수 궤적`의 정답 상태로 확정하지 않는다.
- 군집과 고장이 같은 row에서 연관됐다는 사실을 고장 발생 전 경고 성능으로 표현하지 않는다.
- 논문의 `workload`, `movement speed`, `gripping force`를 공개 CSV의 직접 feature나 알려진 cycle 라벨로 취급하지 않는다.
- 논문의 RTDE 125 Hz 설명을 공개 CSV의 실제 저장 간격으로 사용하지 않는다. CSV의 양수 Timestamp 간격 중앙값은 약 1.005초다.

## 3. 학술연구에서 변경한 분석 기준

| 구분 | 기존 프로젝트 | 학술연구 확장 |
| --- | --- | --- |
| 주 태스크 | 회귀·분류·군집의 폭넓은 비교 | 센서 시계열 구간의 이상탐지에 집중 |
| 평가 단위 | 개별 row | `cycle_run`과 acquisition block |
| 시간 맥락 | 개별 시점 입력 | cycle 경계를 넘지 않는 5·10·20-step window |
| 특징 표현 | 해당 시점 센서값과 파생변수 | mean, std, min, max, range, delta, slope 또는 raw sequence |
| 타깃 | 통합 `System_Failure` 중심 | 통합 타깃과 `ProtectiveStop`, `GripLost` 개별 분석 |
| 분할 | Row random stratified split | cycle-group split과 9개 block held-out 평가 |
| 평가 | Row-level F1, ROC-AUC 중심 | Event cycle recall, 완전 정상 오경보율, 교차 고장 경보율, PR-AUC, block 최악값 |
| 딥러닝 | 향후 과제 | 1D CNN, LSTM 분류기, 정상-only LSTM Autoencoder 비교 |
| 고장 전 예측 | 구분되지 않음 | First positive 이전 window만 쓰는 별도 pre-failure 실험 |

`cycle_run`을 도입한 이유는 원본 `cycle` ID 224, 225, 226, 227, 229, 230, 231, 232, 233이 시간상 떨어진 두 시행 구간에 재등장하기 때문이다. 이를 합치면 실제로 이어지지 않은 구간을 하나의 cycle로 취급하거나 train/test group을 잘못 구성할 수 있다.

## 4. 학술연구에서 실제로 얻은 결과

### 4.1 시계열 구간 특징은 row baseline보다 유용했다

Cycle-group 기준 `SMOTE + Random Forest` 비교 결과는 다음과 같다.

| 타깃 | Row Macro F1 | Best window | Window Macro F1 | Row PR-AUC | Window PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `System_Failure` | 0.7901 | 5-step | 0.8166 | 0.6430 | 0.8613 |
| `ProtectiveStop` | 0.8428 | 5-step | 0.8642 | 0.7409 | 0.9039 |
| `GripLost` | 0.7623 | 10-step | 0.8412 | 0.5604 | 0.8493 |

세 타깃 모두 Macro F1과 PR-AUC가 개선됐다. 이는 짧은 구간의 평균 수준뿐 아니라 변화량과 변동성이 row 한 시점보다 유용할 수 있음을 보여준다. Window 크기는 타깃별로 달랐으므로 하나의 길이가 모든 고장 유형에 최적이라고 결론 내리지는 않는다.

### 4.2 엄격한 block 평가에서도 구간 탐지는 유지됐다

9개 acquisition block을 하나씩 test로 제외한 10-step×19-sensor 동일 입력 비교에서 Random Forest의 핵심 cycle 지표는 다음과 같다. 기존 결과의 `normal_cycle_false_alarm_rate`는 해당 target이 없는 모든 cycle을 분모로 사용했으므로 아래에서는 `target-negative 경보율`로 바로잡았다.

| 타깃 | Event cycle recall | Target-negative 경보율 | 완전 정상 오경보율 | 교차 고장 경보율 | 가장 어려운 block의 recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `System_Failure` | 0.9770 | 0.0435 | 0.0435 | 해당 없음 | 0.9500 |
| `ProtectiveStop` | 0.9516 | 0.0286 | 0.0348 | 0.0000 | 0.5000 |
| `GripLost` | 0.9231 | 0.0368 | 0.0174 | 0.0833 | 0.6667 |

전체 평균은 높지만 `ProtectiveStop`과 `GripLost`의 block 최솟값은 낮다. 따라서 모든 수집 조건에서 동일하게 안정적이라고 주장하지 않는다.

### 4.3 딥러닝이 기본 모델을 개선하지는 못했다

| 타깃 | 모델 | Event cycle recall | 완전 정상 오경보율 | 교차 고장 경보율 |
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

1D CNN은 event recall을 대체로 유지했지만 완전 정상 오경보가 크게 늘었고, 특히 `GripLost`에서는 다른 고장 cycle의 43.75%에도 경보했다. 정상 cycle만 학습한 LSTM Autoencoder는 q95에서 대부분의 event를 탐지하지 못했고, q90으로 낮춰도 결론이 바뀌지 않았다. 따라서 현재 데이터에서는 모델 복잡도를 높이는 것보다 짧은 시계열을 통계 특징으로 정제하는 방식이 더 안정적이다.

### 4.4 구간 탐지와 고장 전 예측은 다른 태스크였다

구간 탐지는 window 안에 이미 positive 상태가 포함될 수 있다. 이를 고장 전 예측과 분리하기 위해 first positive 이전의 데이터만 사용한 pre-failure 실험을 수행했다.

- `GripLost`: 30회 반복 split에서 positive F1 평균 `0.3058`, 적어도 하나의 positive를 탐지한 split 비율 `0.9667`
- `System_Failure`: positive F1 평균 `0.0920`
- `ProtectiveStop`: positive F1 평균 `0.0867`

`GripLost`에서 약한 사전 신호 가능성은 남았지만 절대 성능과 분산을 고려하면 실용적인 조기경고 모델을 확보했다고 볼 수 없다.

### 4.5 공정조건은 공개 파일만으로 복원할 수 없었다

논문에는 `workload`, `movement speed`, `gripping force`의 세 수준이 기술돼 있지만, 공개 CSV와 Excel에는 cycle-to-condition 대응표가 없다. 센서 기반 3-cluster도 실제 세 수준과 일치하지 않았고, 25-cycle block은 온도만으로 매우 잘 구분됐다. 이는 공정조건 차이와 thermal/session drift가 혼재할 가능성을 보여준다.

따라서 현재 block은 일반화 성능을 시험하는 수집 구간 proxy로만 사용한다. 실제 공정조건 분류는 대응표를 확보하기 전까지 연구 결과에 포함하지 않는다.

### 4.6 센서 패턴 해석은 탐색적 결과다

오경보 window에서는 일부 전류·속도·온도·`Tool_current`의 수준 또는 변화 범위 차이가 반복됐다. 그러나 겹치는 window의 사후 기술통계이며 모델별 오류에 선택된 표본이다. 이를 feature importance, 통계적 독립 표본 검정, 물리적 인과관계로 확대하지 않는다.

### 4.7 센서 그룹 ablation은 전류 중심 결론을 좁혔다

결과 확인 전에 센서 그룹 10개 변형과 해석 규칙을 고정하고, 동일한 10-step·9-block·Random Forest 조건에서 270회 평가했다. `all_19` 결과는 기존 `12` 기준선의 score와 prediction에 정확히 일치했다.

- 전류 계열을 모두 제거하면 `System_Failure` event recall은 `0.9770`에서 `0.9310`으로 낮아지고 완전 정상 오경보율은 `0.0435`에서 `0.1217`로 증가했다.
- `GripLost` event recall은 전류 계열 제거 시 `0.9231`에서 `0.4103`으로 크게 낮아졌다.
- `ProtectiveStop`은 전류 계열 제거 시 recall이 `0.9516`에서 `0.9677`로 높아졌지만, 완전 정상 오경보율 `0.0609`와 교차 고장 경보율 `0.1600`으로 tradeoff가 나타났다.
- `Tool_current`만 제거하면 `System_Failure`와 `ProtectiveStop`은 recall이 소폭 높아지고 완전 정상 오경보율은 유지됐다. `GripLost`는 recall과 완전 정상 오경보율이 같았지만 교차 고장 경보율은 `0.0833`에서 `0.1042`로 높아졌다.
- 속도 단독 입력은 높은 recall과 높은 오경보가 함께 나타났고, 온도 단독 입력은 세 타깃 모두 recall이 낮았다. 다만 온도 제거 후 일부 타깃의 오경보가 증가해 온도는 수집 구간 차이를 보정하는 데 기여했을 가능성이 있다.

따라서 관절 전류 계열이 특히 `GripLost` 탐지에 기여한다는 해석은 지지되지만, 전류 전체가 항상 또는 유일하게 핵심이라는 결론은 지지되지 않는다. 이 결과는 센서 그룹의 예측 기여에 대한 제한적 검증이며 물리적 인과관계의 증거가 아니다.

### 4.8 기본 분류 모델은 신호를 잡았지만 오경보를 억제하지 못했다

동일한 133개 시계열 통계 특징과 9개 block-held-out 평가에서 Logistic Regression과 RBF SVM을 tuning 없이 비교했다. Random Forest의 고정 결과를 공통 기준선으로 재사용했다.

| Target | 모델 | Event cycle recall | 완전 정상 오경보율 | 교차 고장 경보율 |
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

Logistic Regression의 높은 recall은 완전 정상 cycle의 27.0~53.0%와 다른 고장 cycle의 28.0~58.3%에 경보하는 결과와 함께 나타났다. 이는 통계 특징에 선형적으로 포착되는 이상 신호가 있지만 선형 경계만으로 정상 구간과 고장 유형을 안정적으로 구분하지 못했음을 보여준다. RBF SVM은 Logistic Regression보다 경보를 줄였지만 `System_Failure`에서는 Random Forest보다 두 지표가 모두 불리했고, 나머지 타깃에서는 recall과 경보 부담의 trade-off가 남았다. 현재 조건에서는 Random Forest의 비선형 분기와 특징 상호작용 처리가 추가적인 균형에 기여한 것으로 해석한다.

## 5. 이 확장이 학술연구로서 갖는 가치

### 5.1 새 모델보다 평가 문제를 바로잡았다

핵심 기여는 새로운 architecture 제안이 아니다. 시계열 데이터에서 무엇을 독립 표본으로 보고, 어느 경계를 넘지 않게 하며, 어떤 단위로 오경보를 계산할지를 명시했다. 이는 기존 row random split 결과가 답하지 못한 일반화 질문을 다룬다.

### 5.2 성공과 실패를 같은 기준에서 비교했다

동일한 19개 센서와 10-step, 동일한 9개 outer test block을 사용해 Random Forest, 1D CNN, LSTM Autoencoder를 비교했다. Autoencoder의 threshold와 해석 규칙은 결과 확인 전에 고정했다. 딥러닝이 열세였다는 결과도 변경하거나 제외하지 않았다.

### 5.3 운영상 의미가 있는 지표를 추가했다

겹치는 window의 평균 점수만으로 성능을 주장하지 않고, 실제 고장 cycle을 하나라도 잡았는지와 완전 정상 cycle 및 다른 고장 cycle에서 경보가 누적되는지를 함께 계산했다. 이를 통해 1D CNN의 높은 recall 뒤에 완전 정상 오경보와 교차 고장 경보 부담이 있음을 확인했다.

### 5.4 기존 프로젝트의 강한 표현을 검증 가능한 주장으로 좁혔다

후속 연구를 통해 다음과 같이 표현 범위를 조정할 수 있다.

- `고장을 예측했다`보다 `이상이 포함된 센서 구간을 탐지했다`가 정확하다.
- `전류가 고장의 원인이다`보다 `전류 기반 특징이 반복적으로 유용했다`가 정확하다.
- `과부하 군집이 조기경고를 가능하게 한다`보다 `특정 전류 군집에서 Protective Stop이 더 자주 관찰됐다`가 정확하다.
- `딥러닝으로 성능을 개선했다`가 아니라 `딥러닝이 기본 모델보다 불리한 조건을 확인했다`가 실제 결과다.

### 5.5 사후 센서 해석을 사전 고정 비교로 보완했다

오류 window 기술통계에서 보인 센서 차이를 그대로 결론으로 삼지 않고, 센서 그룹과 평가 기준을 먼저 고정한 뒤 제거·단독 입력 비교를 수행했다. 결과가 타깃과 오경보 지표에 따라 달랐기 때문에 `전류가 중요하다`는 단일 문장 대신 어떤 전류 그룹이 어느 타깃에서 기여했는지와 반례를 함께 보고할 수 있게 됐다.

### 5.6 특징 설계와 모델 구조의 효과를 분리했다

동일 시계열 특징을 선형 모델, kernel 모델, 트리 앙상블에 입력해 모델만 바꾸는 통제 비교를 수행했다. 기본 모델도 event 신호를 포착했지만 정상 cycle 오경보 차이가 컸다. 이를 통해 성능 향상을 시계열 특징 정제만의 효과 또는 Random Forest만의 효과로 단순화하지 않고, 두 요소의 결합으로 설명할 수 있다.

## 6. 현재 연구의 한계

1. 공개된 단일 UR3 CobotOps 수집자료만 사용했으며 외부 데이터 검증이 없다.
2. 동일 센서 비교의 cycle-level 평가 표본은 202개다. 4,035개 겹치는 window를 독립 표본으로 볼 수 없다.
3. Acquisition block은 실제 공정조건 라벨이 아니라 cycle 번호 기반 proxy다.
4. Block/session 내부 상관과 thermal drift가 남아 있어 Wilson 신뢰구간만으로 모든 불확실성을 설명할 수 없다.
5. Timestamp 간격이 완전히 균일하지 않아 일반적인 고주파 시계열로 가정할 수 없다.
6. 구간 탐지의 positive window에는 이미 발생한 이상 상태가 포함될 수 있다.
7. Pre-failure positive 표본이 적어 `ProtectiveStop`과 통합 `System_Failure`의 사전 예측 근거가 약하다.
8. Autoencoder calibration 정상 cycle은 fold별 13-36개로 quantile 임계값의 해상도가 제한된다.
9. Hyperparameter는 최소 비교 범위로 사전 고정했으며 광범위한 architecture 탐색이나 통계적 최적화를 수행하지 않았다.
10. Random Forest는 `random_state=42`의 고정 1회이고 딥러닝은 3개 seed consensus다. 딥러닝 초기화 변동은 확인했지만 Random Forest의 seed 변동은 반복 평가하지 않았다.
11. Random Forest feature importance와 오류 window의 센서 차이는 인과관계가 아니다.
12. 센서 그룹 제거·단독 사용은 입력 feature 수와 SMOTE가 작동하는 공간을 함께 바꾸므로, 관찰된 성능 차이를 개별 센서의 인과적 효과로 해석할 수 없다.
13. Logistic Regression과 RBF SVM은 고정 `C=1`의 대표 설정 한 개만 비교했으며, 각 모델 계열의 최적 성능이나 통계적 동등성을 검증한 것은 아니다.
14. SVM probability와 Random Forest probability는 산출 방식이 달라 score 자체를 보정된 확률로 직접 비교할 수 없다.
15. 실제 로봇 환경의 실시간 지연, 연속 경보 정책, 새로운 고장 유형 대응은 검증하지 않았다.

## 7. 연구계획서 이행 상태

| 계획 항목 | 상태 | 현재 판단 |
| --- | --- | --- |
| 데이터 감사와 baseline 재현 | 완료 | UCI 원본 provenance와 결측·라벨·cycle 구조 확인 |
| 시계열 구간 재구성과 특징 생성 | 완료 | 5·10·20-step과 7개 통계량 비교 |
| Row와 시계열 구간 비교 | 완료 | 세 타깃에서 Macro F1·PR-AUC 개선 확인 |
| 1D CNN 비교 | 완료 | 유사한 recall과 높은 오경보 확인 |
| LSTM/GRU Autoencoder | 최소 범위 완료 | LSTM Autoencoder 1개를 정상-only 방식으로 검증; GRU는 추가하지 않음 |
| 고장 유형별 분석 | 완료 | 통합·Protective Stop·GripLost 분리 |
| 오탐·미탐 사례 분석 | 완료 | Seed 반복성, block 집중도, 모델 간 오류 겹침 분석 |
| 센서 패턴 해석 | 제한적 완료 | 오류 기술통계와 사전 고정 센서 그룹 ablation으로 예측 기여를 검증했으나 인과 분석은 아님 |
| 기본 분류 모델 통제 비교 | 완료 | 동일 133개 특징에서 Logistic Regression·RBF SVM·Random Forest 비교 |
| 고장 전조 탐색 | 부분 완료 | GripLost에서 약한 가능성, 조기경고 성능은 미확립 |
| 최종 보고서 | 초안 완료 | 논문 형식 초안을 작성했고 핵심 수치·본문 122개 항목의 내부 대조를 완료함. 전체 재실행·공동 검토·최종 윤문은 남아 있음 |
| 발표자료 | 미완료 | 보고서 내용 확정 후 별도 구성할 산출물 |

## 8. 최종 보고서에 사용할 연구 구조

### 주 연구 질문

> Cycle 및 수집 구간 경계를 보존한 UR3 센서 시계열에서, 짧은 구간의 통계 특징과 raw sequence 기반 모델은 이상 event 탐지, 완전 정상 오경보, 교차 고장 경보 측면에서 어떤 차이를 보이는가?

### 주 결과

- 10-step×19-sensor 통계 특징 기반 `SMOTE + Random Forest`
- 9개 block held-out 평가
- Event cycle recall, 완전 정상 cycle 오경보율, 교차 고장 경보율

### 비교 결과

- 동일 raw sequence의 1D CNN
- 정상-only LSTM Autoencoder
- 딥러닝이 기본 모델을 개선하지 못한 이유와 오류 패턴
- 사전 고정 센서 그룹 ablation을 통한 관절 전류 기여와 반례
- 동일 통계 특징의 Logistic Regression·RBF SVM 통제 비교

### 부가 결과

- `GripLost` 중심 pre-failure 가능성과 한계
- 공정조건 대응표 부재 및 thermal/session drift 진단

### 사용하지 않을 주장

- 실시간 조기경고 시스템을 완성했다는 주장
- 실제 1/2/3 kg, 60/80/100%, 80/100/120 N 조건을 복원했다는 주장
- 딥러닝이 성능을 향상했다는 주장
- 특정 센서가 고장의 원인이라는 주장
- 현재 결과가 다른 로봇이나 공정에 바로 일반화된다는 주장

## 9. 다음 최소 작업

새 모델을 늘리지 않고 제출 가능한 보고서로 확정하기 위해 다음 순서로 진행한다.

1. 공동연구자와 초안의 주 결과·직접 비교·보조 분석 위계를 확정한다.
2. `13`·`14`·`16`에서 본문에 남길 표와 그림을 선정한다.
3. 깨끗한 별도 worktree에서 CPU 파이프라인을 전체 재실행하고 결과를 대조한다.
4. 외부 PyTorch 환경에서 딥러닝 3개 seed를 다시 실행해 cycle prediction과 핵심 지표를 대조한다.
5. 내용 확정 후 수치와 기술 용어를 보존하면서 한국어 문장을 절별로 윤문한다.
6. 참고문헌 형식을 통일하고 최종 PDF의 표·페이지·글꼴을 확인한다.
7. 현재 결과를 근거로 추가 Grid Search, threshold tuning, 주 기준선 교체를 하지 않는다.

## 직접 근거

- 기존 프로젝트: `reports/머신러닝 최종 프로젝트 보고서.pdf`, `reports/Data_Classification_Task_Report.pdf`, `reports/Data_Clustering_Task_Report.pdf`
- 기존 분할 코드: `notebooks/03_5_Classification_RandomForest.ipynb`
- 데이터 감사와 baseline: `research/outputs/00_data_audit_report.md`, `research/outputs/01_baseline_results.md`
- Window 비교: `research/outputs/02_row_vs_window_comparison.csv`
- Pre-failure: `research/outputs/04_pre_failure_repeated_split_summary.csv`, `research/outputs/05_pre_failure_threshold_sensitivity_summary.csv`
- 공정조건 진단: `research/outputs/06_process_condition_separability.md`
- Block 및 event 평가: `research/outputs/08_block_held_out_robustness.md`, `research/outputs/09_event_level_block_validation.md`
- Deep learning 및 오류 분석: `research/outputs/10_sequence_model_comparison.md`, `research/outputs/11_sequence_error_analysis.md`
- 동일 센서 최종 비교: `research/2026-08-23_lstm_autoencoder_preregistration.md`, `research/outputs/12_matched_lstm_autoencoder_comparison.md`
- 최종 평가표: `research/outputs/13_final_evaluation_tables.md`, `research/outputs/13_cycle_consensus_confusion_metrics.csv`
- 센서 그룹 ablation: `research/2026-08-23_sensor_group_ablation_preregistration.md`, `research/outputs/14_sensor_group_ablation.md`, `research/outputs/14_sensor_group_ablation_summary.csv`, `research/outputs/14_sensor_group_ablation_paired_errors.csv`
- 기본 분류 모델 비교: `research/2026-08-23_classical_model_comparison_preregistration.md`, `research/outputs/15_classical_model_comparison.md`, `research/outputs/15_classical_model_summary.csv`, `research/outputs/15_classical_model_paired_errors.csv`
- Fault-context 경보 정정: `research/16_fault_context_alert_analysis.py`, `research/outputs/16_fault_context_alert_analysis.md`, `research/outputs/16_fault_context_alert_summary.csv`
- 직접 관련 연구 및 방법론 참고: `research/2026-08-26_related_work_review.md`
- 논문 형식 연구보고서 초안: `research/2026-08-27_research_report_draft.md`
- 재현성 및 검토 계획: `research/2026-08-27_reproducibility_review_plan.md`
- 보고서 수치 대조: `research/outputs/17_report_evidence_validation.md`
