# 연구보고서 재현성 및 검증 계획

## 목적

연구보고서의 수치가 저장된 결과와 일치하는지 확인하는 단계와, 원본 데이터에서 모델을 다시 학습해 같은 결론이 나오는지 확인하는 단계를 구분한다. 기존 기준 결과물을 직접 덮어쓰지 않고 별도 clean worktree에서 검증한다.

## 현재 완료 범위

- 데이터 파일 SHA-256과 행·열 수 기록
- 주요 CPU·PyTorch 패키지 버전 기록
- seed, window, block, 모델 구조, threshold 규칙 기록
- 결과 확인 전 작성한 사전 고정 문서 보존
- 저장된 CSV를 이용한 fault-context 재집계
- `17_report_evidence_validation.py`를 이용한 보고서 핵심 수치 자동 대조

이 단계는 **내부 일관성 검증**이다. 원본부터의 전체 재실행은 아직 완료하지 않았다.

## 검증 단계

### A. 정적 검증

1. Git commit과 branch를 기록한다.
2. 데이터 SHA-256이 기준값과 같은지 확인한다.
3. Python script가 compile되는지 확인한다.
4. 보고서의 표·본문 수치를 `02`, `04`, `13`, `16` CSV와 대조한다.
5. `TODO`, 임시 placeholder, 잘못된 target 명칭을 검색한다.

### B. CPU 재현 검증

별도 clean worktree에서 `00`~`09`, `10_prepare_sequence_data.py`, `12_matched_rf_baseline.py`, `14_sensor_group_ablation.py`, `14_sensor_group_ablation_results.py`, `15_classical_model_comparison.py`, `15_classical_model_comparison_results.py`를 실행한다. GPU 결과가 필요한 후처리 코드는 GPU 재실행 뒤 수행한다. 중간 cache와 prediction 의존성을 확인하고, 실제 실행 순서를 별도 검증 기록에 남긴다.

판정 기준:

- 데이터 감사, cycle/window 개수, label 분포: exact match
- Random Forest와 기본 모델의 discrete prediction: exact match
- 집계 CSV의 정수 count: exact match
- 부동소수점 metric: 절대오차 `1e-6` 이내
- 실행시간과 파일 생성시각: 비교 대상에서 제외

### C. GPU 재현 검증

기록된 외부 PyTorch 환경에서 `10_torch_sequence_models.py`와 `12_matched_torch_models.py`를 실행하여 1D CNN과 LSTM Autoencoder를 seed 42·43·44로 다시 학습한다. 기존 기준 결과와 섞이지 않도록 별도 output directory와 worktree를 사용한다. 학습 후 `10_sequence_model_results.py`, `11_sequence_error_analysis.py`, `12_matched_results.py`, `13_final_evaluation_tables.py`, `16_fault_context_alert_analysis.py`, `17_report_evidence_validation.py`를 순서대로 실행한다.

판정 기준:

- 모델·seed·fold·epoch·threshold 기록의 완전성: exact match
- Cycle consensus prediction: exact match를 우선 확인
- Headline metric: 절대오차 `1e-4` 이내
- Score 차이가 있으나 discrete prediction과 결론이 같으면 환경 차이로 별도 기록
- 결론을 바꾸는 차이가 발생하면 재현 실패로 처리하고 원인을 조사

### D. 보고서 검증

1. 표의 분모와 평가 단위를 명시한다.
2. Window 결과를 독립 표본 4,035개로 표현하지 않는다.
3. `normal_cycle_false_alarm_rate`의 기존 의미를 `target-negative alert rate`로 바로잡는다.
4. 완전 정상 오경보와 교차 고장 경보를 구분한다.
5. 이상 포함 구간 탐지를 pre-failure 또는 RUL로 표현하지 않는다.
6. 공정조건 대응표가 없는 상태에서 물리조건을 복원했다고 주장하지 않는다.
7. Internal block-held-out 결과를 외부 일반화로 표현하지 않는다.

### E. 문체 및 제출물 검토

내용과 수치가 고정된 뒤 한국어 윤문을 수행한다. 윤문 단계에서는 사실·수치·DOI·모델명·평가 단위를 변경하지 않는다.

검토 순서:

1. 연구 질문과 결론의 일치
2. 방법과 결과의 용어 통일
3. 번역투, 과도한 수동 표현, 반복 문장 정리
4. 표·그림 번호와 본문 인용 확인
5. 참고문헌 형식 통일
6. 최종 PDF의 페이지 잘림, 표 분할, 글꼴, 공백 페이지 확인

## 실행 명령

저장된 결과에 대한 내부 대조:

```powershell
python -X utf8 research\17_report_evidence_validation.py
```

재집계부터 확인:

```powershell
python -X utf8 research\13_final_evaluation_tables.py
python -X utf8 research\16_fault_context_alert_analysis.py
python -X utf8 research\17_report_evidence_validation.py
```

전체 CPU·GPU 재현 명령은 clean worktree를 만든 뒤 현재 `research/README.md`의 실행 순서를 기준으로 별도 검증 기록에 남긴다.

## 완료 조건

- 정적·CPU·GPU 검증 결과가 각각 기록되어 있다.
- 불일치 항목은 원인과 허용 여부가 문서화되어 있다.
- 보고서 표와 본문이 검증된 결과 파일을 가리킨다.
- 공동연구자 검토 후 내용이 고정되어 있다.
- 윤문 후 수치와 기술적 의미가 바뀌지 않았음을 재확인한다.
