# test.ps1 — Run the unit test suite
param(
    [string]$Filter  = "",   # substring filter, e.g. "chunker", "store", "index"
    [switch]$Verbose         # show println! output for passing tests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$runArgs = @("test")

if ($Filter)  { $runArgs += $Filter }
if ($Verbose) { $runArgs += "--", "--nocapture" }

Write-Host "==> cargo $($runArgs -join ' ')"
& cargo @runArgs
exit $LASTEXITCODE
