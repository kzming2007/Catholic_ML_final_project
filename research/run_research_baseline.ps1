$ErrorActionPreference = "Stop"

python -X utf8 research\00_data_audit.py
python -X utf8 research\01_baseline_reproduction.py

Write-Host ""
Write-Host "Research baseline outputs saved to research\outputs."
