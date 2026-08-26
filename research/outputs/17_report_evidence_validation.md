# 17 보고서 근거 대조

- 총 검사: 122
- 통과: 122
- 실패: 0

## 출처별 결과

| source | checks | passed | failed |
| --- | --- | --- | --- |
| 02_row_vs_window_comparison.csv | 12 | 12 | 0 |
| 04_pre_failure_repeated_split_summary.csv | 9 | 9 | 0 |
| 13_window_model_summary.csv | 9 | 9 | 0 |
| 16_fault_context_alert_summary.csv | 52 | 52 | 0 |
| 16_fault_context_cycle_results.csv | 2 | 2 | 0 |
| 2026-08-27_research_report_draft.md | 35 | 35 | 0 |
| dataset/ur3_cobotops.csv | 3 | 3 | 0 |

## 실패 항목

_No rows._

## 해석 범위

- 이 검사는 저장된 CSV와 보고서 핵심 수치의 내부 일관성을 확인한다.
- 모델을 다시 학습하지 않으므로 전체 재현 실험을 대신하지 않는다.
- 전체 재현은 별도 clean worktree에서 CPU·GPU 실행을 다시 수행해 비교한다.
