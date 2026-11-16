# Test users only: the chain RelGT (the best model on the holdout week 2026-09-27 -> 10-04; Taiwan time,
# 09-26 16:00 UTC) trained, tuned and refitted on the users of test_input.csv only, instead of every
# eligible cohort user. Same protocol and settings as relgt_chain in runs/ensemble_ho0927.sh (train on
# weeks whose labels end by 09-19, tune on 09-19 -> 09-26, score the holdout week), plus
#   --train-users test --val-users test --final-users test
# One seed (-Seeds 0), run name relgt_chain_tu_ens0926_s<seed> (~30 min on one GPU, plus ~5 min of token
# sampling the first time).
#
# Comparison: experiments\ensemble\run.py keeps only the users every run has, so scoring this run next
# to the reference runs relgt_chain_ens0926_s0-2 (all users; made by runs/ensemble_ho0927.sh) puts both
# on the same test users, holdout and tune week, with the same decision rule (tune-week share).
# Compare seed:relgt_chain_tu_* with seed:relgt_chain_ens0926_s* and bag:relgt_chain. The "own:" rows
# are not comparable: the reference's own decisions were made and scored on all users.
#
# The run's output goes to experiments\gnn\outputs\logs\<name>.log; new lines are echoed to the console
# every -EchoEvery seconds. Ctrl+C stops the python process too. Runs from any folder.
#   .\runs\test_users_ho0927.ps1                 # full run (seed 0) on the least used GPU, then the comparison
#   .\runs\test_users_ho0927.ps1 -Smoke          # pipeline check: 1 epoch of 30 steps, smoke_ folders,
#                                                #   compared with smoke_relgt_chain_ens0926_s0
#   .\runs\test_users_ho0927.ps1 -Gpu 1          # force a GPU
#   .\runs\test_users_ho0927.ps1 -SkipRuns       # only the comparison
param(
    [int]$Gpu = -1,                  # -1 = the GPU with the least memory in use
    [int[]]$Seeds = @(0),
    [string]$HoWeek = "2026-09-26 16:00",   # UTC start of the scored week (Saturday 16:00)
    [string]$Python = "",
    [switch]$Smoke,
    [switch]$SkipRuns,
    [int]$EchoEvery = 30,
    [int]$FreeMiB = 2000             # a GPU with less memory in use than this counts as free (runs\gpu.ps1)
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent   # the repo root: this script is in runs\
Set-Location $root
$script:GpuFreeMiB = $FreeMiB
. (Join-Path $PSScriptRoot "gpu.ps1")

$suf = "ens" + $HoWeek.Substring(5, 5).Replace("-", "")
$hoT = [datetime]::ParseExact($HoWeek, "yyyy-MM-dd HH:mm", [Globalization.CultureInfo]::InvariantCulture)
$tune = $hoT.AddDays(-7).ToString("yyyy-MM-dd HH:mm")
$hoEnd = $hoT.AddDays(7).ToString("yyyy-MM-dd HH:mm")
$pfx = ""; $fast = @()
if ($Smoke) { $pfx = "smoke_"; $Seeds = @(0); $fast = @("--epochs", "1", "--max-steps-per-epoch", "30") }
$logDir = Join-Path $root "experiments\gnn\outputs\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:TQDM_MININTERVAL = "10"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Python: -Python, else the active venv, else .venv-win
if (-not $Python) {
    if ($env:VIRTUAL_ENV) { $Python = "python" }
    elseif (Test-Path ".venv-win\Scripts\python.exe") { $Python = ".venv-win\Scripts\python.exe" }
    else { $Python = "python" }
}
$pyExe = (Get-Command $Python).Source

function Now { Get-Date -Format "yyyy-MM-dd HH:mm:ss" }
function Quote([string]$a) { if ($a -match '[\s"]') { '"' + $a.Replace('"', '\"') + '"' } else { $a } }

function Pick-Gpu {
    # -Gpu: that GPU once it is free; else the least used free GPU (runs\gpu.ps1). Waits while none is free.
    if ($Gpu -ge 0) { return Wait-FreeGpu @($Gpu) }
    return Wait-FreeGpu
}

function Read-New([string]$path, [ref]$pos) {
    # lines appended to a file another process is still writing, since the last call
    try {
        $fs = [System.IO.File]::Open($path, "Open", "Read", "ReadWrite, Delete")
        try {
            if ($fs.Length -le $pos.Value) { return @() }
            $null = $fs.Seek($pos.Value, "Begin")
            $buf = New-Object byte[] ($fs.Length - $pos.Value)
            $n = $fs.Read($buf, 0, $buf.Length)
            $pos.Value += $n
            return ([System.Text.Encoding]::UTF8.GetString($buf, 0, $n) -split "[\r\n]+" | Where-Object { $_.Trim() })
        } finally { $fs.Close() }
    } catch { return @() }
}

$chain = @("--graph-structure", "chain", "--chain-seed", "latest", "--num-seeds", "16")
$testOnly = @("--train-users", "test", "--val-users", "test", "--final-users", "test")

if (-not $SkipRuns) {
    foreach ($s in $Seeds) {
        $name = "${pfx}relgt_chain_tu_${suf}_s$s"
        $log = Join-Path $logDir "$name.log"
        $err = Join-Path $logDir "$name.stderr.log"
        $runArgs = @("experiments\relgt\run.py") + $chain + $testOnly + @("--epochs", "15", "--end", $hoEnd,
                   "--holdout-weeks", "1", "--val-start", $tune, "--val-end", $HoWeek) + $fast +
                   @("--seed", "$s", "--name", $name)
        $g = Pick-Gpu
        $env:CUDA_VISIBLE_DEVICES = "$g"
        $t0 = Get-Date
        $p = Start-Process -FilePath $pyExe -ArgumentList (($runArgs | ForEach-Object { Quote $_ }) -join " ") `
            -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError $err -NoNewWindow -PassThru
        $null = $p.Handle                    # keep a handle, or ExitCode stays empty after the process ends
        Write-Host "$(Now) [gpu$g $name] START (pid $($p.Id)) -> $log"
        $pos = [long]0
        try {
            while (-not $p.HasExited) {
                Start-Sleep -Seconds $EchoEvery
                Read-New $log ([ref]$pos) | ForEach-Object { Write-Host "[gpu$g $name] $_" }
            }
        } finally {                          # Ctrl+C: do not leave python on the GPU
            if (-not $p.HasExited) {
                Write-Host "$(Now) stopping $name (pid $($p.Id))"
                Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            }
        }
        $p.WaitForExit()
        Read-New $log ([ref]$pos) | ForEach-Object { Write-Host "[gpu$g $name] $_" }
        # stderr at the end of the log: warnings / traceback, without the tqdm redraws
        if ((Test-Path $err) -and (Get-Item $err).Length -gt 0) {
            $lines = (Get-Content $err -Raw -Encoding UTF8) -split "\n" |
                ForEach-Object { ($_ -split "\r" | Where-Object { $_.Trim() } | Select-Object -Last 1) } |
                Where-Object { $_ -and $_ -notmatch '\d+%\|' }
            if ($lines) {
                Add-Content -Path $log -Value "`n---- stderr ----" -Encoding UTF8
                $lines | Add-Content -Path $log -Encoding UTF8
            }
        }
        Remove-Item $err -ErrorAction SilentlyContinue
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        if ($p.ExitCode -ne 0) {
            Write-Host "$(Now) [gpu$g $name] FAILED (exit $($p.ExitCode)) after $mins min, see $log"
            Get-Content $log -Tail 5 -Encoding UTF8 | ForEach-Object { Write-Host "    $_" }
            exit 1
        }
        Write-Host "$(Now) [gpu$g $name] DONE after $mins min"
    }
    Write-Host "$(Now) all runs finished"
}

Write-Host "`n== on test users: ${pfx}relgt_chain_tu vs ${pfx}relgt_chain"
& $pyExe experiments\ensemble\run.py --suffix $suf "${pfx}relgt_chain_tu" "${pfx}relgt_chain"
