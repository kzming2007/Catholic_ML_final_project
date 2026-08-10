# 2026-07-08 연구 착수 요약

## 완료한 작업

- `UR3_Cobot_ML/research` 아래에 별도 연구 작업 공간을 만들었다.
- 데이터 감사와 baseline 재현을 위한 재현 가능 스크립트를 추가했다.
- 초기 데이터 감사 결과와 baseline 결과 파일을 `research/outputs` 아래에 생성했다.

## 주요 데이터 감사 결과

- 원본 데이터셋: 7,409 rows x 24 columns.
- 원본 기준 `dropna()` 후 행 수: 7,355.
- 고유 cycle 수: 240.
- `System_Failure = ProtectiveStop OR GripLost`.
- 타깃 불균형:
  - `System_Failure`: ProtectiveStop 결측 행 제외 후 positive 518 rows.
  - `ProtectiveStop`: positive 278 rows.
  - `GripLost`: positive 243 rows.
- 공개 CSV의 Timestamp 간격은 대부분 약 1.0초다.
- 원 논문의 공정 조건인 `workload`, `movement speed`, `gripping force`는 공개 CSV의 직접 컬럼이 아니다.

## Baseline 결과 요약

| Target | Split | Model | Macro F1 | Positive Recall | PR-AUC |
|---|---|---|---:|---:|---:|
| System_Failure | random_stratified | rf_smote | 0.8013 | 0.5865 | 0.6601 |
| System_Failure | cycle_group | rf_smote | 0.7901 | 0.5638 | 0.6430 |
| ProtectiveStop | random_stratified | rf_smote | 0.7921 | 0.6607 | 0.6160 |
| ProtectiveStop | cycle_group | rf_smote | 0.8428 | 0.7111 | 0.7409 |
| GripLost | random_stratified | rf_smote | 0.7988 | 0.5918 | 0.6943 |
| GripLost | cycle_group | rf_smote | 0.7623 | 0.4898 | 0.5604 |

## 해석

- 기존 프로젝트의 Random Forest + SMOTE baseline은 `System_Failure` 기준으로 재현된다.
- `cycle_group` split에서 성능이 크게 무너지지는 않지만, 고장 유형별 결과는 달라진다.
- `ProtectiveStop`과 `GripLost`는 별도 타깃 분석이 필요할 만큼 다른 양상을 보인다.
- 다음 연구 단계는 딥러닝이 아니라 window 단위 표현 추가가 먼저다.

## 다음 단계

작성 대상: `02_window_feature_baseline.py`.

계획 범위:

1. 5, 10, 20 step window를 구성한다.
2. mean, std, min, max, range, first-last delta, simple slope feature를 생성한다.
3. window 내부에 positive target이 하나라도 있는지를 기준으로 window target을 부여한다.
4. cycle-group split 기준으로 Random Forest와 SMOTE + Random Forest를 비교한다.
5. row baseline과 window feature baseline을 비교한다.
