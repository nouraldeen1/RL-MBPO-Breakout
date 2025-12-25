# Run medium Dreamer training and save logs
# Usage: Open PowerShell in repo root and run: .\scripts\run_medium.ps1

if (-not (Test-Path -Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Make sure we run with visible output and save to logs
python .\src\main.py 2>&1 | Tee-Object -FilePath .\logs\medium_run.txt
