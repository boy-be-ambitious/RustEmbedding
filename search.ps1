# search.ps1 — Query an existing index
param(
    [Parameter(Mandatory=$true)]
    [string]$Query,
    [string]$Index  = "index.bin",
    [string]$Model  = "C:\Program Files\Huawei\DevEco Studio\plugins\codegenie-plugin\embedding_model\VESO-model\VESO-25M",
    [string]$Ort    = "C:\Users\qinzh\Downloads\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll",
    [int]   $Top    = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Ort)) {
    Write-Error "onnxruntime.dll not found at '$Ort'. Use -Ort to specify the path."
}

$bin = Join-Path $PSScriptRoot "target\release\rust-embedding.exe"

& $bin search `
    "--query=$Query" `
    "--index=$Index" `
    "--model=$Model" `
    "--ort=$Ort" `
    "--top=$Top"

exit $LASTEXITCODE
