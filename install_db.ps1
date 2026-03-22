# install_db.ps1 — Install LanceDB dependencies and build the project
#
# This script:
#   1. Downloads and installs protoc (required by LanceDB)
#   2. Adds LanceDB dependencies to Cargo.toml
#   3. Builds the project with LanceDB support
#
# Usage:
#   .\install_db.ps1          # Full installation
#   .\install_db.ps1 -SkipProtoc  # Skip protoc installation if already installed

param(
    [switch]$SkipProtoc
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProtocVersion = "29.3"
$ProtocDir = "C:\protoc"
$ProtocBin = Join-Path $ProtocDir "bin\protoc.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LanceDB Installation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Install protoc ─────────────────────────────────────────────────────
if (-not $SkipProtoc) {
    Write-Host "[Step 1/4] Installing protoc..." -ForegroundColor Yellow
    
    if (Test-Path $ProtocBin) {
        Write-Host "  protoc already installed at: $ProtocBin" -ForegroundColor Green
        & $ProtocBin --version
    } else {
        Write-Host "  Downloading protoc $ProtocVersion..."
        $ProtocUrl = "https://github.com/protocolbuffers/protobuf/releases/download/v$ProtocVersion/protoc-$ProtocVersion-win64.zip"
        $ProtocZip = Join-Path $env:TEMP "protoc.zip"
        
        try {
            # Download
            Invoke-WebRequest -Uri $ProtocUrl -OutFile $ProtocZip -UseBasicParsing
            Write-Host "  Downloaded to: $ProtocZip"
            
            # Extract
            if (Test-Path $ProtocDir) {
                Remove-Item $ProtocDir -Recurse -Force
            }
            Expand-Archive -Path $ProtocZip -DestinationPath $ProtocDir -Force
            Write-Host "  Extracted to: $ProtocDir"
            
            # Verify
            if (Test-Path $ProtocBin) {
                Write-Host "  protoc installed successfully!" -ForegroundColor Green
                & $ProtocBin --version
            } else {
                Write-Error "protoc.exe not found after extraction"
            }
            
            # Cleanup
            Remove-Item $ProtocZip -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Error "Failed to install protoc: $_"
        }
    }
} else {
    Write-Host "[Step 1/4] Skipping protoc installation (-SkipProtoc)" -ForegroundColor Gray
}

Write-Host ""

# ── Step 2: Set environment variables ──────────────────────────────────────────
Write-Host "[Step 2/4] Setting environment variables..." -ForegroundColor Yellow

$env:PROTOC = $ProtocBin
$env:PATH = "$ProtocDir\bin;$env:PATH"

Write-Host "  PROTOC = $env:PROTOC"
Write-Host "  PATH updated with protoc bin directory"

# Persist to user environment
[Environment]::SetEnvironmentVariable("PROTOC", $ProtocBin, "User")
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notlike "*$ProtocDir\bin*") {
    [Environment]::SetEnvironmentVariable("PATH", "$ProtocDir\bin;$UserPath", "User")
    Write-Host "  Added protoc to user PATH (persistent)"
}

Write-Host ""

# ── Step 3: Update Cargo.toml ─────────────────────────────────────────────────
Write-Host "[Step 3/4] Updating Cargo.toml for LanceDB..." -ForegroundColor Yellow

$CargoToml = Join-Path $PSScriptRoot "Cargo.toml"
$CargoContent = @'
[package]
name = "rust-embedding"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "rust-embedding"
path = "src/main.rs"

[dependencies]
ort = { version = "2.0.0-rc.10", default-features = false, features = ["load-dynamic", "std", "ndarray", "api-24"] }
ort-sys = { version = "2.0.0-rc.10", default-features = false, features = ["api-24"] }
tokenizers = { version = "0.21", default-features = false, features = ["onig"] }
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
bincode = "1"
rayon = "1"
anyhow = "1"
walkdir = "2"
sha2 = "0.10"
sysinfo = "0.33"
log = "0.4"
env_logger = "0.11"
chrono = { version = "0.4", features = ["serde"] }
parking_lot = "0.12"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
lancedb = "0.4"
lance = "0.19"
arrow-array = "53"
arrow-schema = "53"
futures = "0.3"

[target.'cfg(windows)'.dependencies]
windows = { version = "0.58", features = [
    "Win32_System_ProcessStatus",
    "Win32_System_Threading",
    "Win32_Foundation",
    "Win32_System_LibraryLoader",
    "Win32_Storage_FileSystem",
] }

[profile.release]
opt-level = 3
lto = "thin"
'@

Set-Content -Path $CargoToml -Value $CargoContent -Encoding UTF8
Write-Host "  Cargo.toml updated with LanceDB dependencies"

Write-Host ""

# ── Step 4: Build the project ─────────────────────────────────────────────────
Write-Host "[Step 4/4] Building project with LanceDB..." -ForegroundColor Yellow
Write-Host "  This may take 10-30 minutes on first build (500+ dependencies)" -ForegroundColor Gray
Write-Host ""

$BuildStart = Get-Date

& cargo build --release 2>&1 | ForEach-Object {
    if ($_ -match "error") {
        Write-Host "  $_" -ForegroundColor Red
    } elseif ($_ -match "Compiling") {
        Write-Host "  $_" -ForegroundColor DarkGray
    } elseif ($_ -match "Finished") {
        Write-Host "  $_" -ForegroundColor Green
    } else {
        Write-Host "  $_"
    }
}

$BuildEnd = Get-Date
$BuildDuration = $BuildEnd - $BuildStart

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  LanceDB Installation Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Build duration: $($BuildDuration.ToString('mm\:ss'))" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Next steps:" -ForegroundColor Cyan
    Write-Host "    1. Run: .\build.ps1 -Db index.lance" -ForegroundColor White
    Write-Host "    2. Or update build.ps1 default -Db to 'index.lance'" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Build Failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Check the error messages above" -ForegroundColor Red
    exit 1
}
