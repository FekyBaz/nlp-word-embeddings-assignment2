$ErrorActionPreference = "Stop"

# Run from any location; all paths resolve relative to the repository root.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $RepoRoot "data/raw"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "Kaggle CLI was not found. Install it with: py -m pip install kaggle"
}

# Never place Kaggle credentials in this repository. Configure them using
# https://www.kaggle.com/settings/api (or supported environment variables).
kaggle datasets download -d "sunilthite/text-document-classification-dataset" -p $DataDir --unzip

if ($LASTEXITCODE -ne 0) {
    throw "Kaggle download failed with exit code $LASTEXITCODE. Confirm authentication and dataset access."
}

Write-Host "Download completed. Files in data/raw:"
Get-ChildItem -Path $DataDir -Recurse -File | ForEach-Object {
    "{0} ({1:N0} bytes)" -f $_.FullName, $_.Length
}
Write-Host ""
Write-Host "Next: inspect file schema, labels, license/terms, and data quality before analysis."
