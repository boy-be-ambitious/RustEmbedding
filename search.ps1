# search.ps1 — Query an existing index.db
#
# Default paths are read from config.txt (key=value, one per line).
param(
    [Parameter(Mandatory=$true)]
    [string]$Query,
    [string]$Index = "index.db",
    [string]$Model = "",   # filled from config.txt if empty
    [string]$Ort   = "",   # filled from config.txt if empty
    [int]   $Top   = 5
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
if (-not (Test-Path $Index)) { Write-Error "Index database not found at '$Index'. Run build.ps1 first." }

# ── Run ───────────────────────────────────────────────────────────────────────
$bin = Join-Path $PSScriptRoot "target\release\rust-embedding.exe"

& $bin search `
    "--query=$Query" `
    "--db=$Index" `
    "--model=$Model" `
    "--ort=$Ort" `
    "--top=$Top"

exit $LASTEXITCODE
