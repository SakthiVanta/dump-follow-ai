# One-shot install for Windows — ensures opencv GUI build wins over headless
Set-Location "$PSScriptRoot\.."

if (-not (Test-Path "venv")) {
    python -m venv venv
}

.\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
# albumentations installs opencv-python-headless (no GUI) — override it:
pip install opencv-python --force-reinstall --no-deps

Write-Host ""
Write-Host "Install complete. Run:  python main.py demo" -ForegroundColor Green
