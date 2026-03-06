Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.12 and ensure py.exe is on PATH."
}

& py -3.12 -c "import sys; print(sys.version)" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 is required. Install it first (for example: winget install -e --id Python.Python.3.12)."
}

if (Test-Path ".venv") {
    Remove-Item -Recurse -Force ".venv"
}

& py -3.12 -m venv .venv

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -e .

Write-Host "Bootstrap complete."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
