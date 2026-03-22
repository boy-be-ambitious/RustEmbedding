# test.ps1 — Run the unit test suite
param(
    [string]$Filter  = "",
    [switch]$Verbose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$args_list = @("test")

if ($Filter) {
    $args_list += $Filter
}

if ($Verbose) {
    $args_list += "--", "--nocapture"
}

Write-Host "==> cargo $($args_list -join ' ')"
& cargo @args_list
exit $LASTEXITCODE
