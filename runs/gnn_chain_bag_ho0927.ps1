# Seed bagging of the GNN on the chain graph (gnn_chain) on the holdout week 2026-09-27 -> 10-04 (Taiwan time;
# 09-26 16:00 UTC), with RelGT's two stages:
#   validation model: trained on weeks whose labels end by 09-19, early stopping and the decision shares on
#                     the tune week 09-19 -> 09-26 (as gnn_chain in runs/chain_ho0927.sh);
#   final model:      refit on every labelled week through 09-26 (--holdout-final), scored with the same shares.
# Epochs of both stages come from the recorded best epoch of the same seed's documented run
# (experiments\gnn\outputs\gnn_chain_ho0926_s<seed>\metrics.json: 1 on seeds 0-2), so no stage trains epochs
# that early stopping would throw away. -Epochs / -FinalEpochs override them.
# Runs gnn_chain_ens0926_s<seed> (seeds 0-2, one at a time on one GPU; token sampling ~5 min a seed the first
# time, then ~1 min of training per stage), then experiments\ensemble\run.py:
#   bag of gnn_chain (validation models, then final models), and gnn_chain + relgt_chain (relgt_chain_ens0926_s0-2,
#   made by runs/ensemble_ho0927.sh) with both final models, push by the tune-week share and by prev-week.
# The run's output goes to experiments\gnn\outputs\logs\<name>.log, echoed every -EchoEvery seconds.
#   .\runs\gnn_chain_bag_ho0927.ps1                  # full run: seeds 0-2, then the ensembles
#   .\runs\gnn_chain_bag_ho0927.ps1 -Smoke           # pipeline check: seed 0, 1 epoch of 30 steps, smoke_ folders
#   .\runs\gnn_chain_bag_ho0927.ps1 -SkipRuns        # only the ensembles
#   .\runs\gnn_chain_bag_ho0927.ps1 -Gpu 1 -Seeds 0,1,2,3,4
param(
    [int]$Gpu = -1,                  # -1 = the GPU with the least memory in use
    [int[]]$Seeds = @(0, 1, 2),
    [int]$Epochs = 0,                # 0 = the recorded best epoch of gnn_chain_ho0926_s<seed>
    [int]$FinalEpochs = 0,           # 0 = the same
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

$md = $HoWeek.Substring(5, 5).Replace("-", "")
$suf = "ens$md"
$pfx = ""; $fast = @()
if ($Smoke) { $pfx = "smoke_"; $Seeds = @(0); $Epochs = 1; $FinalEpochs = 1; $fast = @("--max-steps-per-epoch", "30") }
$logDir = Join-Path $root "experiments\gnn\outputs\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:TQDM_MININTERVAL = "10"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

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

function Recorded-BestEpoch([int]$seed) {
    # best epoch of the documented gnn_chain run of this seed; else the largest one recorded for any seed
    $f = "experiments\gnn\outputs\gnn_chain_ho${md}_s$seed\metrics.json"
    if (Test-Path $f) { return [int](Get-Content $f -Raw | ConvertFrom-Json).best_epoch }
    $all = Get-ChildItem "experiments\gnn\outputs\gnn_chain_ho${md}_s*\metrics.json" -ErrorAction SilentlyContinue |
        ForEach-Object { [int](Get-Content $_.FullName -Raw | ConvertFrom-Json).best_epoch }
    if ($all) { return ($all | Measure-Object -Maximum).Maximum }
    throw "no recorded gnn_chain_ho${md}_s* run: pass -Epochs and -FinalEpochs"
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

if (-not $SkipRuns) {
    $i = 0
    foreach ($s in $Seeds) {
        $i++
        $name = "${pfx}gnn_chain_${suf}_s$s"
        $ep = if ($Epochs -gt 0) { $Epochs } else { Recorded-BestEpoch $s }
        $fep = if ($FinalEpochs -gt 0) { $FinalEpochs } else { $ep }
        $log = Join-Path $logDir "$name.log"
        $err = Join-Path $logDir "$name.stderr.log"
        $runArgs = @("experiments\gnn\run.py") + $chain + @("--holdout", "--holdout-week", $HoWeek, "--holdout-final",
                   "--epochs", "$ep", "--final-epochs", "$fep") + $fast + @("--seed", "$s", "--name", $name)
        $g = Pick-Gpu
        $env:CUDA_VISIBLE_DEVICES = "$g"
        $t0 = Get-Date
        $p = Start-Process -FilePath $pyExe -ArgumentList (($runArgs | ForEach-Object { Quote $_ }) -join " ") `
            -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError $err -NoNewWindow -PassThru
        $null = $p.Handle                    # keep a handle, or ExitCode stays empty after the process ends
        Write-Host "$(Now) [gpu$g $name] START ($i/$($Seeds.Count), epochs $ep, final epochs $fep, pid $($p.Id)) -> $log"
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

function Ens([string[]]$a) {
    Write-Host "`n== ensemble: $($a -join ' ')"
    & $pyExe experiments\ensemble\run.py --suffix $suf @a
}
$gc = "${pfx}gnn_chain"; $rc = "${pfx}relgt_chain"
Ens @($gc)                                              # bag of the validation models
Ens @($gc, "--gnn-variant", "final")                    # bag of the final models
Ens @($gc, $rc, "--gnn-variant", "final")               # + RelGT chain, both final models
Ens @($gc, $rc, "--gnn-variant", "final", "--push-rule", "prev-week")   # the submission's push rule
