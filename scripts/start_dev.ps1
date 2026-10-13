<#
.SYNOPSIS
    Ses Analizi Platformu - Geliştirme Ortamı Başlatma Betiği (PowerShell)
.DESCRIPTION
    1. Sanal ortam (.venv) kontrolü ve aktivasyonu
    2. Redis bağlantı kontrolü (çalışmıyorsa docker compose ile başlatma girişimi)
    3. Port 8000 çakışma denetimi (WinError 10013 önleme, PID tespiti)
    4. API (uvicorn) ve Stream Worker'ı eşzamanlı / ayrı pencerelerde başlatma
#>

[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$HostAddress = "127.0.0.1",
    [switch]$NoWorker,
    [switch]$SingleWindow
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Ses Analizi Platformu - Geliştirme Ortamı Başlatılıyor   " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Sanal Ortam (.venv) Kontrolü
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $VenvPython)) {
    Write-Error "HATA: .venv sanal ortamı bulunamadı ($VenvPython). Lütfen önce 'python -m venv .venv' ve paket kurulumlarını yapın."
    exit 1
}

Write-Host "[1/4] Sanal ortam doğrulandı: .venv" -ForegroundColor Green
& $VenvActivate

# 2. Redis Kontrolü
$UseRedisStream = $env:USE_REDIS_STREAM
if ($null -eq $UseRedisStream -or $UseRedisStream -eq "") {
    $UseRedisStream = "true"
}

if ($UseRedisStream -eq "true") {
    Write-Host "[2/4] Redis bağlantısı kontrol ediliyor (127.0.0.1:6379)..." -ForegroundColor Yellow
    $RedisTest = Test-NetConnection -ComputerName "127.0.0.1" -Port 6379 -InformationLevel Quiet -WarningAction SilentlyContinue
    if (-not $RedisTest) {
        Write-Warning "Redis 6379 portunda çalışmıyor!"
        Write-Host "Docker Compose ile Redis başlatılmaya çalışılıyor ('docker compose up -d redis')..." -ForegroundColor Yellow
        try {
            docker compose up -d redis
            Start-Sleep -Seconds 3
            $RedisRetry = Test-NetConnection -ComputerName "127.0.0.1" -Port 6379 -InformationLevel Quiet -WarningAction SilentlyContinue
            if ($RedisRetry) {
                Write-Host "Redis başarıyla Docker üzerinde başlatıldı." -ForegroundColor Green
            } else {
                Write-Warning "Redis Docker ile başlatılamadı. Stream Worker kullanıyorsanız Redis gereklidir."
                Write-Warning "Redis olmadan çalıştırmak için .env dosyasında USE_REDIS_STREAM=false ayarlayabilirsiniz."
            }
        } catch {
            Write-Warning "Docker çalıştırılamadı. Lütfen Redis'i elle başlatın veya USE_REDIS_STREAM=false kullanın."
        }
    } else {
        Write-Host "[2/4] Redis çalışıyor (127.0.0.1:6379)." -ForegroundColor Green
    }
} else {
    Write-Host "[2/4] USE_REDIS_STREAM=false aktif: Redis gerekmiyor (Doğrudan in-memory / thread pool modu)." -ForegroundColor Green
}

# 3. Port Çakışması Kontrolü (WinError 10013 Önleme)
Write-Host "[3/4] Port $Port kullanılabilirlik kontrolü yapılıyor..." -ForegroundColor Yellow
$ExistingConnections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue

if ($ExistingConnections) {
    $ConflictingPid = $ExistingConnections[0].OwningProcess
    $ProcessInfo = Get-Process -Id $ConflictingPid -ErrorAction SilentlyContinue
    $ProcName = if ($ProcessInfo) { $ProcessInfo.ProcessName } else { "Bilinmeyen" }

    Write-Error "HATA: Port $Port zaten PID $ConflictingPid ($ProcName) tarafından kullanılıyor!"
    Write-Host "WinError 10013 çakışmasını önlemek için lütfen süreci sonlandırın:" -ForegroundColor Red
    Write-Host "  Stop-Process -Id $ConflictingPid -Force" -ForegroundColor Red
    Write-Host "veya betiği farklı bir port ile çalıştırın: .\scripts\start_dev.ps1 -Port 8001" -ForegroundColor Yellow
    exit 1
}
Write-Host "[3/4] Port $Port boş ve kullanıma hazır." -ForegroundColor Green

# 4. API ve Worker Başlatma
Write-Host "[4/4] Servisler başlatılıyor..." -ForegroundColor Cyan

if ($SingleWindow) {
    Write-Host "Tek pencere modu: Yalnızca API başlatılıyor (Worker için ayrı terminalde 'python -m audio_analyzer.workers.stream_worker' çalıştırabilirsiniz)." -ForegroundColor Yellow
    & $VenvPython -m uvicorn audio_analyzer.api.main:app --host $HostAddress --port $Port --reload
} else {
    # Worker'ı ayrı pencerede başlat (eğer stream modu açıksa ve --NoWorker verilmemişse)
    if ($UseRedisStream -eq "true" -and -not $NoWorker) {
        Write-Host "Stream Worker ayrı bir PowerShell penceresinde başlatılıyor..." -ForegroundColor Green
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ProjectRoot'; & '$VenvActivate'; Write-Host '--- STREAM WORKER ---' -ForegroundColor Cyan; python -m audio_analyzer.workers.stream_worker"
    }

    Write-Host "API Uvicorn servisi başlatılıyor (http://${HostAddress}:${Port})..." -ForegroundColor Green
    & $VenvPython -m uvicorn audio_analyzer.api.main:app --host $HostAddress --port $Port --reload
}
