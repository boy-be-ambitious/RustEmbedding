# build.ps1 — Compile the project and incrementally index a repo into index.db
#
# Default paths are read from config.txt (key=value, one per line).
# Any parameter can be overridden on the command line.
param(
    [string]$Repo        = "hmosworld-master",
    [string]$Model       = "",        # filled from config.txt if empty
    [string]$Ort         = "",        # filled from config.txt if empty
    [string]$Database    = "",        # defaults to <Repo>.db if empty
    [int]   $Batch       = 16,
    [string]$Report      = "report",
    [switch]$SkipCompile,
    [switch]$Force        # wipe the DB and re-embed everything from scratch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Load config.txt defaults ──────────────────────────────────────────────────
$configFile = Join-Path $PSScriptRoot "config.txt"
if (Test-Path $configFile) {
    Get-Content $configFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+?)\s*=\s*(.+?)\s*$') {
            switch ($Matches[1]) {
                "VESO_ONNX"   { if (-not $Model) { $Model = $Matches[2] } }
                "ONNXRUNTIME" { if (-not $Ort)   { $Ort   = $Matches[2] } }
            }
        }
    }
}

# ── Validate ──────────────────────────────────────────────────────────────────
if (-not $Model) { Write-Error "Model path not set. Add VESO_ONNX=<path> to config.txt or pass -Model." }
if (-not $Ort)   { Write-Error "ORT path not set. Add ONNXRUNTIME=<path> to config.txt or pass -Ort." }
if (-not (Test-Path $Ort))   { Write-Error "onnxruntime.dll not found at '$Ort'." }
if (-not (Test-Path $Model)) { Write-Error "Model directory not found at '$Model'." }

# ── Compile ───────────────────────────────────────────────────────────────────
if (-not $SkipCompile) {
    Write-Host "==> cargo build --release"
    cargo build --release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ── Run ───────────────────────────────────────────────────────────────────────
$bin = Join-Path $PSScriptRoot "target\release\rust-embedding.exe"

if (-not $Database) {
    $Database = "$Repo.db"
}

Write-Host "==> Building index for '$Repo' -> '$Database' ..."

$runArgs = @(
    "build",
    "--repo=$Repo",
    "--model=$Model",
    "--ort=$Ort",
    "--db=$Database",
    "--batch=$Batch",
    "--report=$Report"
)
if ($Force) { $runArgs += "--force" }

& $bin @runArgs
exit $LASTEXITCODE
