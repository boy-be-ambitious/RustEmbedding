# build.ps1 — Compile the project, index a repo, write report.json / report.md
param(
    [string]$Repo   = "hmosworld-master",
    [string]$Model  = "C:\Program Files\Huawei\DevEco Studio\plugins\codegenie-plugin\embedding_model\VESO-model\VESO-25M",
    [string]$Ort    = "C:\Users\qinzh\Downloads\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll",
    [string]$Out    = "index.bin",
    [int]   $Batch  = 16,
    [string]$Report = "report",
    [switch]$SkipCompile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Locate onnxruntime.dll ────────────────────────────────────────────────────
if (-not (Test-Path $Ort)) {
    Write-Error "onnxruntime.dll not found at '$Ort'. Use -Ort to specify the path."
}

# ── Compile ───────────────────────────────────────────────────────────────────
if (-not $SkipCompile) {
    Write-Host "==> cargo build --release"
    cargo build --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ── Run ───────────────────────────────────────────────────────────────────────
$bin = Join-Path $PSScriptRoot "target\release\rust-embedding.exe"

Write-Host "==> Building index for '$Repo' ..."

& $bin build `
    "--repo=$Repo" `
    "--model=$Model" `
    "--ort=$Ort" `
    "--out=$Out" `
    "--batch=$Batch" `
    "--report=$Report"

exit $LASTEXITCODE
