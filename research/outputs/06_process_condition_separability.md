# 06 공정조건 분리 가능성 진단

## 연구 질문

공개 UR3 CobotOps 센서 데이터만 사용해 논문에 제시된 `movement speed`, `workload`, `gripping force`의 서로 다른 수준을 cycle별로 독립 구분할 수 있는지 탐색한다.

## 논문 근거와 증거 한계

- 논문은 세 정적 공정조건을 `workload` 1/2/3 kg, `movement speed` 60/80/100%, `gripping force` 80/100/120 N으로 제시하고 각 test scenario를 25회 반복했다고 기술한다.
- 여기서 hyperparameter는 학습률이나 tree 깊이 같은 ML hyperparameter가 아니라 로봇 실험의 고정 공정조건이다.
- 논문은 가능한 27개 조합을 모두 사용했는지, 각 scenario가 어느 cycle 범위에 해당하는지 공개하지 않는다.
- 공개 Excel/CSV에는 세 공정조건 라벨이나 scenario 조합표가 없고, `Cycle`은 시행 번호만 제공한다.
- 원본 Excel에는 보이는 `data` 시트 하나만 있으며, 숨은 시트·숨은 열·defined name·cell comment·조건 관련 문자열도 확인되지 않았다.
- 따라서 실제 조건 수준에 대한 supervised accuracy는 계산할 수 없다. 아래 결과는 센서에 3개 latent regime이 보이는지와 25-cycle acquisition block이 구분되는지를 진단한 간접 근거다.
- 논문: Tyrovolas et al. (2024), DOI 10.1007/978-3-031-63851-0_6.

## 전처리

- 원본 `cycle` 값이 비연속적으로 재등장하는 ID: [224, 225, 226, 227, 229, 230, 231, 232, 233].
- 동일 cycle ID의 재등장 구간을 합치지 않고, 시간순 연속 구간마다 별도 `cycle_run`을 부여했다.
- 고장 자체가 조건 분리를 대신하지 않도록 각 시행의 first positive 이전 healthy prefix만 요약했다.
- healthy prefix가 5개 미만인 시행 71개는 제외했다.
- cycle ID나 Timestamp는 입력 feature로 사용하지 않았다.

## 25-cycle 후보 블록

| scenario_block_25 | cycle_runs | included_cycle_runs | cycle_min | cycle_max | faults | complete_like |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 25 | 23 | 1 | 25 | 7 | True |
| 2 | 25 | 22 | 26 | 50 | 5 | True |
| 3 | 24 | 21 | 51 | 75 | 6 | True |
| 4 | 25 | 19 | 76 | 100 | 13 | True |
| 5 | 25 | 22 | 101 | 125 | 6 | True |
| 6 | 4 | 3 | 126 | 150 | 2 | False |
| 7 | 25 | 19 | 151 | 175 | 9 | True |
| 8 | 24 | 17 | 176 | 200 | 13 | True |
| 9 | 27 | 12 | 201 | 225 | 21 | True |
| 10 | 31 | 13 | 226 | 250 | 24 | True |
| 11 | 14 | 7 | 251 | 264 | 9 | False |

## 공정조건별 3-cluster 진단

| parameter_proxy | cycle_runs | feature_count | kmeans_k | silhouette | davies_bouldin | repeat_ari_mean | gmm_bic_best_components_1to5 | gmm_bic_k1 | gmm_bic_k2 | gmm_bic_k3 | gmm_bic_k4 | gmm_bic_k5 | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| movement_speed | 178 | 30 | 3 | 0.2044 | 1.6 | 0.9052 | 5 | 15465.1705 | 13864.2163 | 12453.4161 | 11678.581 | 10977.8202 | 3개 수준 구조를 지지하기 어려움 |
| workload | 178 | 30 | 3 | 0.4037 | 1.471 | 0.9419 | 5 | 15465.1705 | 13145.718 | 12818.0665 | 12776.4116 | 12742.023 | 분리는 보이나 3개 수준과 일치하지 않음 |
| gripping_force | 178 | 5 | 3 | 0.2728 | 1.1704 | 0.9058 | 5 | 2577.5284 | 1935.0119 | 1671.8017 | 1547.5521 | 1512.3101 | 분리는 보이나 3개 수준과 일치하지 않음 |

### Cluster별 물리 대리변수 범위

| parameter_proxy | cluster_level | cycle_runs | proxy_mean | proxy_std | proxy_min | proxy_max |
| --- | --- | --- | --- | --- | --- | --- |
| movement_speed | 1 | 52 | 0.2746 | 0.0512 | 0.1754 | 0.5104 |
| movement_speed | 2 | 61 | 0.4885 | 0.143 | 0.2595 | 0.7485 |
| movement_speed | 3 | 65 | 0.6244 | 0.0958 | 0.2489 | 0.711 |
| workload | 1 | 154 | 1.3751 | 0.1632 | 0.8583 | 2.0723 |
| workload | 2 | 16 | 1.6416 | 0.203 | 1.3289 | 2.2415 |
| workload | 3 | 8 | 1.6868 | 0.1868 | 1.4654 | 2.0277 |
| gripping_force | 1 | 52 | 0.1686 | 0.0254 | 0.0901 | 0.2332 |
| gripping_force | 2 | 110 | 0.1765 | 0.0321 | 0.0965 | 0.3139 |
| gripping_force | 3 | 16 | 0.3832 | 0.0513 | 0.3207 | 0.4901 |

- `movement_speed`는 joint speed 요약, `workload`는 joint current 요약, `gripping_force`는 `Tool_current` 요약으로 진단했다.
- Silhouette은 cluster 간 분리도, `repeat_ari_mean`은 초기값을 바꿨을 때 cluster 재현성, GMM BIC는 데이터가 선호하는 component 수를 뜻한다.
- GMM component 수는 1-5 범위에서만 비교했다.
- cluster level 1/2/3은 대리변수 평균의 낮음/중간/높음 순서일 뿐 실제 60/80/100%, 1/2/3 kg, 80/100/120 N 라벨이 아니다.

### Cluster 결과 해석

- `movement_speed`: Silhouette 0.2044, GMM 최적 component 5개로, 세 속도 수준이 자연스럽게 분리된다는 근거가 약하다.
- `workload`: Silhouette 0.4037로 일부 분리는 보이지만 GMM 최적 component는 5개이고 cluster 크기가 154/16/8로 치우쳤다. 1/2/3 kg의 균형 잡힌 세 수준으로 해석할 수 없다.
- `gripping_force`: Silhouette 0.2728, GMM 최적 component 5개이며 낮음·중간 cluster의 proxy 범위가 겹친다. 80/100/120 N을 독립 복원했다고 볼 수 없다.

## 25-cycle 후보 블록 분류

| feature_set | model | cycle_runs | scenario_blocks | cv_splits | balanced_accuracy_mean | balanced_accuracy_std | macro_f1_mean | macro_f1_std | permutation_pvalue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| process_sensors | dummy_most_frequent | 168 | 9 | 4 | 0.1111 | 0.0 | 0.0267 | 0.0018 |  |
| process_sensors | logistic_regression | 168 | 9 | 4 | 0.206 | 0.0358 | 0.1994 | 0.0387 | 0.0099 |
| process_sensors | random_forest | 168 | 9 | 4 | 0.1866 | 0.0259 | 0.1712 | 0.0332 |  |
| temperature_only | dummy_most_frequent | 168 | 9 | 4 | 0.1111 | 0.0 | 0.0267 | 0.0018 |  |
| temperature_only | logistic_regression | 168 | 9 | 4 | 0.594 | 0.0443 | 0.5909 | 0.0492 | 0.0099 |
| temperature_only | random_forest | 168 | 9 | 4 | 0.962 | 0.0327 | 0.9608 | 0.0352 |  |

- 논문의 25회 반복을 근거로 cycle ID를 25개씩 묶고, 20개 이상 시행이 남은 블록만 사용했다.
- 높은 분류 성능은 acquisition block 사이에 센서 분포 차이가 있음을 뜻한다. 공정조건 조합의 차이인지 시간 경과, 온도 drift, 재설정 같은 session effect인지는 라벨 없이 분리할 수 없다.
- `temperature_only`가 높은 성능을 보이면 공정조건보다 시간·장비 상태가 block 구분에 기여했을 가능성을 함께 고려해야 한다.
- 실제로 process sensor Logistic Regression의 balanced accuracy는 0.2060로 chance 0.1111보다 높지만 낮은 수준이었다. 반면 temperature-only Random Forest는 0.9620로 매우 높아, block 정체성이 thermal/session drift에 강하게 남아 있음을 보여준다.

## 결론 사용 범위

- 직접 결론: 공개 파일만으로 세 공정조건의 실제 값을 cycle별로 확정하거나 조건 분류 정확도를 검증할 수 없다.
- 간접 결론: 공정 sensor에는 25-cycle block 차이가 약하게 남지만, 세 물리조건의 3개 수준과 일치하는 단순 구조는 확인되지 않았다. block 구분은 온도와 수집 순서의 영향을 크게 받는다.
- 금지할 해석: 현재 cluster를 1/2/3 kg, 60/80/100%, 80/100/120 N의 정답 라벨로 간주해 후속 supervised model을 학습하면 순환논증이 된다.
- 후속 검증: 저자 또는 원 수집팀의 cycle-to-condition mapping을 확보하면, `cycle_run` 단위 group split으로 세 조건을 각각 supervised classification하여 실제 구분 가능성을 검증한다.
