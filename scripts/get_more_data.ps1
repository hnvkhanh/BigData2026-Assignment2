# Download earlier GH Archive hours (cohort-only), then rebuild data/db from everything.
#
#   powershell -ExecutionPolicy Bypass -File scripts\get_more_data.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\get_more_data.ps1 -Start 2024-12-01 -Workers 16
#
# Re-runnable: hours already in data\gharchive are skipped, so an interrupted run resumes.
param(
    [string]$Start = "2025-01-01",   # first hour to fetch, UTC
    [string]$End = "2025-12-01",     # stop before this hour (where the current data begins)
    [int]$Workers = 16,               # parallel hour downloads + filters
    [switch]$SkipProcess             # download only
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Raw = Join-Path $Root "data\gharchive"
$Out = Join-Path $Root "data"
$Db = Join-Path $Out "db"

Write-Host "== Step 1: download $Start -> $End into $Raw"
New-Item -ItemType Directory -Force $Raw | Out-Null
$startUtc = [datetime]::SpecifyKind([datetime]::Parse($Start), "Utc")
$endUtc = [datetime]::SpecifyKind([datetime]::Parse($End), "Utc")
$totalHours = [int]($endUtc - $startUtc).TotalHours
$baseline = @(Get-ChildItem $Raw -Filter "*.json.gz").Count
$todo = $totalHours
# Hours already on disk inside the range are skipped by the downloader.
for ($h = $startUtc; $h -lt $endUtc; $h = $h.AddHours(1)) {
    if (Test-Path (Join-Path $Raw ("{0:yyyy-MM-dd}-{1}.json.gz" -f $h, $h.Hour))) { $todo-- }
}
Write-Host "$totalHours hours in range, $todo to fetch"

$sw = [Diagnostics.Stopwatch]::StartNew()
# The downloader retries each hour for ~18 min (rides out the nightly internet outage). If any
# hour still fails it exits with code 2; wait and run it again (finished hours are skipped).
$MaxPasses = 10
for ($pass = 1; $pass -le $MaxPasses; $pass++) {
    if ($pass -gt 1) { Write-Host "Some hours failed (network down?). Waiting 60 s, then pass $pass of $MaxPasses..."; Start-Sleep -Seconds 60 }
    $proc = Start-Process python -NoNewWindow -PassThru -ArgumentList @(
        "scripts\download_gharchive.py", "--out", "`"$Raw`"", "--cohort-only",
        "--start", $Start, "--end", $End, "--workers", $Workers)
    $null = $proc.Handle   # PS 5.1: without caching the handle, ExitCode comes back empty
    while (-not $proc.HasExited) {
        $done = @(Get-ChildItem $Raw -Filter "*.json.gz").Count - $baseline
        if ($todo -gt 0) {
            $pct = [math]::Min(100, [int](100 * $done / $todo))
            $rate = if ($sw.Elapsed.TotalSeconds -gt 10 -and $done -gt 0) { $done / $sw.Elapsed.TotalMinutes } else { 0 }
            $eta = if ($rate -gt 0) { "{0:N0} min left" -f (($todo - $done) / $rate) } else { "estimating..." }
            Write-Progress -Activity "Downloading GH Archive" -PercentComplete $pct `
                -Status ("{0}/{1} hours ({2}%), {3:N1} hours/min, {4}" -f $done, $todo, $pct, $rate, $eta)
        }
        Start-Sleep -Seconds 3
    }
    $proc.WaitForExit()
    if ($proc.ExitCode -eq 0) { break }
    if ($proc.ExitCode -ne 2) { throw "download crashed (exit code $($proc.ExitCode))" }
}
Write-Progress -Activity "Downloading GH Archive" -Completed
if ($proc.ExitCode -ne 0) { throw "download still failing after $MaxPasses passes; re-run the script later" }
Write-Host ("Download finished in {0:N1} min" -f $sw.Elapsed.TotalMinutes)

if ($SkipProcess) { Write-Host "Skipping processing."; exit 0 }

if (Test-Path $Db) {
    $Backup = Join-Path $Out ("db_backup_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
    Write-Host "== Backing up existing tables to $Backup"
    Copy-Item $Db $Backup -Recurse
}

Write-Host "== Step 2: preprocess all hours in $Raw -> $Db"
python assignment2\scripts\preprocess.py --raw $Raw --out $Out
if ($LASTEXITCODE -ne 0) { throw "preprocess failed" }

Write-Host "== Done. Tables are in $Db"
