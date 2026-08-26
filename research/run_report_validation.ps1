$ErrorActionPreference = "Stop"

python -X utf8 research\13_final_evaluation_tables.py
python -X utf8 research\16_fault_context_alert_analysis.py
python -X utf8 research\17_report_evidence_validation.py

Write-Host ""
Write-Host "Report evidence validation completed."
