# 2026-08-23 센서 그룹 ablation 사전 고정 기록

## 문서 목적

기존 수업 프로젝트에서 반복적으로 제시된 `전류 계열 변수가 고장 탐지에 중요하다`는 관찰이 cycle 및 수집 block 경계를 보존한 평가에서도 유지되는지 확인한다. 전체 결과 실행 전에 입력 그룹, 비교 조합, 모델, 분할, 지표와 해석 규칙을 고정한다.

이 문서는 정식 외부 사전등록이 아니라 Git 커밋으로 시점을 보존하는 내부 사전 고정 기록이다.

## 연구 질문

> 동일한 10-step window와 9개 block-held-out 평가에서 joint current, speed, temperature, Tool current 센서 그룹을 제거하거나 단독 사용했을 때 Random Forest의 이상 event 탐지율과 정상 cycle 오경보율은 어떻게 달라지는가?

이 실험의 목적은 최고 성능 feature 조합을 사후 탐색하는 것이 아니다. 기존 프로젝트의 전류 중심 해석이 엄격한 분할에서도 관찰되는지 검증하고, 각 센서 그룹이 타깃별로 제공하는 정보의 범위를 기술하는 것이다.

## 데이터와 평가 표본

- 데이터: UCI UR3 CobotOps 공개 데이터셋.
- 입력: `cycle_run` 경계를 넘지 않는 10-step window.
- 대상 block: `1, 2, 3, 4, 5, 7, 8, 9, 10`.
- 평가 cycle: 202개.
  - `System_Failure`: event 87개, normal 115개.
  - `ProtectiveStop`: event 62개, normal 140개.
  - `GripLost`: event 39개, normal 163개.
- Window: 4,035개. Window가 겹치므로 독립 표본 수로 해석하지 않는다.
- 타깃: `System_Failure`, `ProtectiveStop`, `GripLost`를 각각 평가한다.

## 센서 그룹

파생변수 `Power_J0`-`Power_J5`와 `Abs_Current_Sum`은 제외한다. 전류 정보가 파생변수를 통해 다른 그룹에 중복 유입되는 것을 막고, `12`와 같은 19개 원본 센서를 사용한다.

| 그룹 | 센서 | 개수 |
| --- | --- | ---: |
| `joint_current` | `Current_J0`-`Current_J5` | 6 |
| `temperature` | `Temperature_J0`-`Temperature_J5` | 6 |
| `speed` | `Speed_J0`-`Speed_J5` | 6 |
| `tool_current` | `Tool_current` | 1 |
| `current_family` | `joint_current` + `tool_current` | 7 |

각 선택 센서에서 mean, std, min, max, range, first-last delta, slope 7개 통계량을 계산한다.

## 고정 비교 조합

모든 조합을 보고하며 결과가 좋은 조합만 선택하지 않는다.

| Variant | 포함 센서 | 목적 |
| --- | --- | --- |
| `all_19` | 전체 19개 | 고정 기준선 |
| `current_family_only` | joint current + Tool current | 전류 계열 단독 정보량 |
| `joint_current_only` | joint current | 관절 전류 단독 정보량 |
| `tool_current_only` | Tool current | 말단 전류 단독 정보량 |
| `speed_only` | speed | 속도 단독 정보량 |
| `temperature_only` | temperature | 온도 단독 정보량과 drift 가능성 |
| `drop_current_family` | speed + temperature | 전류 계열 제거 영향 |
| `drop_speed` | joint current + temperature + Tool current | 속도 제거 영향 |
| `drop_temperature` | joint current + speed + Tool current | 온도 제거 영향 |
| `drop_tool_current` | joint current + temperature + speed | Tool current 제거 영향 |

## 모델과 분할

- 모델: `SMOTE + Random Forest`.
- Trees: 300.
- `random_state=42`.
- SMOTE `k_neighbors=min(5, minority_count-1)`.
- Decision rule: `score > 0.50`.
- Outer 평가: 9개 block을 한 번씩 test로 사용하고 나머지 8개 block으로 학습한다.
- Scaling은 사용하지 않는다.
- Feature 조합별 hyperparameter tuning은 수행하지 않는다.
- `all_19` 결과는 기존 `12_matched_rf_window_predictions.csv`와 prediction 및 score가 일치해야 한다. 불일치하면 ablation 결과를 해석하기 전에 원인을 확인한다.

## 평가 지표

### Primary cycle 지표

- Event cycle recall: 실제 positive window가 있는 cycle에서 positive window를 하나 이상 탐지한 비율.
- Normal cycle false-alarm rate: 실제 positive window가 없는 cycle에서 하나 이상의 window가 positive로 예측된 비율.
- 두 비율의 Wilson 95% confidence interval.
- Held-out block별 event recall 최솟값과 정상 cycle false-alarm rate 최댓값.

### Secondary window 지표

- Accuracy.
- Macro F1.
- Positive precision, recall, F1.
- ROC-AUC.
- PR-AUC.
- Confusion matrix.

### Paired cycle 오류 비교

각 variant를 `all_19`와 같은 cycle에서 비교한다.

- 양쪽이 함께 놓친 event.
- Variant에서 새로 놓친 event와 새로 복구한 event.
- 양쪽이 함께 낸 정상 cycle 오경보.
- Variant에서 새로 발생한 오경보와 해소된 오경보.

## 해석 규칙

### 제거 실험: necessity

- `drop_X`가 `all_19`보다 event recall이 낮거나 같고 정상 cycle 오경보율이 높거나 같으며, 두 지표 중 하나 이상이 불리하면 해당 그룹이 현재 모델에 기여했다는 기술적 근거로 해석한다.
- Recall과 오경보율 방향이 엇갈리면 trade-off로 보고 단일 우위를 주장하지 않는다.
- 제거 후 좋아지면 해당 그룹이 현재 block 일반화에서 교란 또는 불필요한 정보로 작용했을 가능성을 제시한다. 원인으로 확정하지 않는다.

### 단독 실험: sufficiency

- `current_family_only`, `joint_current_only`, `tool_current_only`, `speed_only`, `temperature_only`는 `all_19`와의 절대 차이를 모두 보고한다.
- 사전에 정하지 않은 허용 오차를 적용해 `충분하다`고 이분법적으로 판정하지 않는다.
- 단독 모델의 우수한 결과는 해당 그룹만으로도 구분 정보가 남는다는 뜻이며 다른 그룹이 불필요하다는 인과적 결론은 아니다.

### 기존 전류 중심 결론

- 타깃별로 `current_family_only`와 다른 단독 그룹을 비교하고, `drop_current_family`와 `all_19`의 paired cycle 오류를 함께 본다.
- 세 타깃 결과가 다르면 통합된 `전류가 항상 핵심`이라는 결론 대신 고장 유형별 차이로 보고한다.
- Random Forest feature importance의 순위와 ablation 성능 저하는 서로 다른 근거다. Ablation도 센서 그룹의 인과적 효과를 증명하지 않는다.

## 고정 출력

- `research/14_sensor_group_ablation.py`
- `research/14_sensor_group_ablation_results.py`
- `research/outputs/14_sensor_group_ablation_predictions.csv`
- `research/outputs/14_sensor_group_ablation_runs.csv`
- `research/outputs/14_sensor_group_ablation_window_metrics.csv`
- `research/outputs/14_sensor_group_ablation_cycle_results.csv`
- `research/outputs/14_sensor_group_ablation_block_metrics.csv`
- `research/outputs/14_sensor_group_ablation_summary.csv`
- `research/outputs/14_sensor_group_ablation_paired_errors.csv`
- `research/outputs/14_sensor_group_ablation.md`

## 해석 제한

- Feature 수는 variant마다 달라진다. 제거 효과에는 센서 정보뿐 아니라 차원 수와 SMOTE 공간의 변화도 포함된다.
- 같은 물리량 안의 센서는 상관돼 있으므로 센서 하나의 독립적 기여를 분리하는 실험이 아니다.
- `Tool_current`는 이상 상태가 발생한 뒤 변할 수 있다. 구간 탐지에서의 유용성을 고장 전조로 해석하지 않는다.
- Temperature는 block/session drift를 반영할 수 있으므로 높은 성능이 직접적인 고장 메커니즘을 뜻하지 않는다.
- Random Forest는 고정 seed 1회다. 모델 seed 변동에 대한 반복 검증은 포함하지 않는다.
- 9개 block은 실제 공정조건 정답이 아니라 수집 구간 proxy다.
- 동일 공개 데이터에 대한 내부 분석이며 외부 검증이 아니다.
- 이 결과를 보고 feature 조합, threshold, window 크기 또는 모델 hyperparameter를 바꾸지 않는다.
