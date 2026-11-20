# All cohort users and no filler tokens: the chain RelGT (holdout week 2026-09-27 -> 10-04; Taiwan time, 09-26 16:00 UTC)
# trained and refitted on every cohort user of each week (9,000, no eligibility exclusion; runs/relgt_pool_ho0927.ps1),
# and with the sampler stopping when a user has no neighbour left, instead of filling the K = 100 token slots.
# Same protocol and settings as relgt_chain in runs/ensemble_ho0927.sh (train on weeks whose labels end by 09-19,
# tune on 09-19 -> 09-26, score the holdout week), plus
#   --train-pool cohort --pad-tokens
#
# What --pad-tokens changes (experiments/relgt/run.py): by default an example with fewer than K tokens keeps all of its
# real tokens and fills the remaining slots with RANDOM REPEATS of them (a user with some neighbours), or with random
# visible nodes of the graph (a user with no neighbour at all). With --pad-tokens the remaining slots are padding:
#   - tokens: the slot keeps the fallback hop as its marker and no subgraph edges;
#   - attention: no token attends to a padding slot (-inf key bias);
#   - readout: padding slots get zero weight (a user with no neighbour reads out its seed token only).
# Most of the ~3,000 users added by --train-pool have almost no history, so this is where the default fill matters.
# The token cache key changes (a "pad" entry), so the tokens are sampled again for every week (~10 min once per seed).
#
# Three seeds (-Seeds 0,1,2), run name relgt_chain_poolpad_ens0926_s<seed>; a seed that already has holdout.npz is
# skipped (seed 0 exists, so only seeds 1 and 2 train; -Force reruns them). The ensemble step then bags the seeds
# (bag:relgt_chain_poolpad, mean of the seeds' rank probabilities).
#
# Jobs = seeds, queued over -Gpus (default 0 and 1): each GPU runs one job at a time and takes the next when it is
# free: less than -FreeMiB (2000) MiB in use, so a GPU that another program is using is skipped until it frees up
# (runs\gpu.ps1). With two missing seeds and two GPUs both train at the same time (~30-60 min; the padding mask makes
# attention slower per step). Every seed samples its tokens itself (the cache key contains the seed; ~8-10 min).
#
# Comparison: experiments\ensemble\run.py scores this run next to the reference runs relgt_chain_ens0926_s0-2 (eligible
# users, default fill) and, if it exists, relgt_chain_pool_ens0926_s* (all users, default fill): on the same holdout
# and tune-week users with the same decision rule (tune-week share). Compare bag:relgt_chain_poolpad with
# bag:relgt_chain_pool (the effect of padding) and with bag:relgt_chain (both changes), 3 seeds each.
#
# Each job's output goes to experiments\gnn\outputs\logs\<name>.log; new lines of every running job are echoed every
# -EchoEvery seconds. Ctrl+C stops the running jobs too. Runs from any folder.
#   .\runs\relgt_poolpad_ho0927.ps1              # seeds 0-2 (the missing ones) on GPUs 0 and 1, then the comparison
#   .\runs\relgt_poolpad_ho0927.ps1 -Gpus 1      # on GPU 1 only (-Gpu 1 does the same)
#   .\runs\relgt_poolpad_ho0927.ps1 -Smoke       # pipeline check: 1 epoch of 30 steps, smoke_ folders
#   .\runs\relgt_poolpad_ho0927.ps1 -SkipRuns    # only the comparison
param(
    [int[]]$Gpus = @(0, 1),
    [int]$Gpu = -1,                  # one GPU only (same as -Gpus <n>)
    [int[]]$Seeds = @(0, 1, 2),
    [switch]$Force,                  # retrain seeds that already have a holdout.npz
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
if ($Gpu -ge 0) { $Gpus = @($Gpu) }

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

$chain = @("--graph-structure", "chain", "--chain-seed", "latest", "--num-seeds", "16")
$common = @("experiments\relgt\run.py") + $chain + @("--train-pool", "cohort", "--pad-tokens", "--epochs", "15",
           "--end", $hoEnd, "--holdout-weeks", "1", "--val-start", $tune, "--val-end", $HoWeek) + $fast

function Now { Get-Date -Format "yyyy-MM-dd HH:mm:ss" }
function Quote([string]$a) { if ($a -match '[\s"]') { '"' + $a.Replace('"', '\"') + '"' } else { $a } }

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

function Start-Run($job, [int]$gpu) {
    $log = Join-Path $logDir "$($job.Name).log"
    $err = Join-Path $logDir "$($job.Name).stderr.log"
    $runArgs = $common + @("--seed", "$($job.Seed)", "--name", $job.Name)
    $env:CUDA_VISIBLE_DEVICES = "$gpu"   # inherited by the new process
    $p = Start-Process -FilePath $pyExe -ArgumentList (($runArgs | ForEach-Object { Quote $_ }) -join " ") `
        -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError $err -NoNewWindow -PassThru
    $null = $p.Handle                    # keep a handle, or ExitCode stays empty after the process ends
    Write-Host "$(Now) [gpu$gpu $($job.Name)] START ($script:started/$total, pid $($p.Id)) -> $log"
    @{ Job = $job; Proc = $p; Log = $log; Err = $err; Start = Get-Date; Pos = [long]0 }
}

function Echo-Run($run, [int]$gpu) {
    $pos = $run.Pos
    Read-New $run.Log ([ref]$pos) | ForEach-Object { Write-Host "[gpu$gpu $($run.Job.Name)] $_" }
    $run.Pos = $pos
}

function Complete-Run($run, [int]$gpu) {
    $run.Proc.WaitForExit()
    Echo-Run $run $gpu
    # stderr at the end of the log: warnings / traceback, without the tqdm redraws
    if ((Test-Path $run.Err) -and (Get-Item $run.Err).Length -gt 0) {
        $lines = (Get-Content $run.Err -Raw -Encoding UTF8) -split "\n" |
            ForEach-Object { ($_ -split "\r" | Where-Object { $_.Trim() } | Select-Object -Last 1) } |
            Where-Object { $_ -and $_ -notmatch '\d+%\|' }
        if ($lines) {
            Add-Content -Path $run.Log -Value "`n---- stderr ----" -Encoding UTF8
            $lines | Add-Content -Path $run.Log -Encoding UTF8
        }
    }
    Remove-Item $run.Err -ErrorAction SilentlyContinue
    $mins = [math]::Round(((Get-Date) - $run.Start).TotalMinutes, 1)
    if ($run.Proc.ExitCode -eq 0) {
        $score = Select-String -Path $run.Log -Encoding UTF8 -Pattern '^mean holdout score' | Select-Object -Last 1
        Write-Host "$(Now) [gpu$gpu $($run.Job.Name)] DONE after $mins min  $($score.Line)"
    } else {
        Write-Host "$(Now) [gpu$gpu $($run.Job.Name)] FAILED (exit $($run.Proc.ExitCode)) after $mins min, see $($run.Log)"
        Get-Content $run.Log -Tail 5 -Encoding UTF8 | ForEach-Object { Write-Host "    $_" }
        $script:failed++
    }
}

if (-not $SkipRuns) {
    $queue = New-Object System.Collections.Queue
    foreach ($s in $Seeds) {
        $name = "${pfx}relgt_chain_poolpad_${suf}_s$s"
        if (-not $Force -and (Test-Path "experiments\relgt\outputs\$name\holdout.npz")) {
            Write-Host "$(Now) [$name] already done (holdout.npz exists), skipped; -Force reruns it"
            continue
        }
        $queue.Enqueue(@{ Seed = $s; Name = $name })
    }
    $total = $queue.Count
    Write-Host "$(Now) $total jobs on GPUs $($Gpus -join ', ')"
    $running = @{}
    $busy = @{}                          # gpu -> true while another program holds it (logged once)
    $script:started = 0; $script:failed = 0
    try {
        while ($queue.Count -gt 0 -or $running.Count -gt 0) {
            foreach ($g in $Gpus) {
                if ($running.ContainsKey($g) -and $running[$g].Proc.HasExited) {
                    Complete-Run $running[$g] $g
                    $running.Remove($g)
                    Start-Sleep -Seconds 5   # let the finished job release its memory before the check
                }
                if (-not $running.ContainsKey($g) -and $queue.Count -gt 0) {
                    # only a GPU that no other program is using: memory in use under -FreeMiB
                    $used = (Get-GpuMemoryUsed)[$g]
                    if ($used -ge $script:GpuFreeMiB) {
                        if (-not $busy[$g]) {
                            Write-Host "$(Now) gpu$g busy ($used MiB >= $($script:GpuFreeMiB) MiB): waiting for it"
                            $busy[$g] = $true
                        }
                        continue
                    }
                    if ($busy[$g]) { Write-Host "$(Now) gpu$g free again ($used MiB)"; $busy[$g] = $false }
                    $script:started++
                    $running[$g] = Start-Run $queue.Dequeue() $g
                }
            }
            Start-Sleep -Seconds $EchoEvery
            foreach ($g in @($running.Keys)) { Echo-Run $running[$g] $g }
        }
    } finally {                          # Ctrl+C: do not leave python processes on the GPUs
        foreach ($r in $running.Values) {
            if (-not $r.Proc.HasExited) {
                Write-Host "$(Now) stopping $($r.Job.Name) (pid $($r.Proc.Id))"
                Stop-Process -Id $r.Proc.Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
    Write-Host "$(Now) all $total jobs finished, $script:failed failed"
}

# This run, the all-users run with the default fill (if it exists) and the eligible-users reference.
$cfgs = @("${pfx}relgt_chain_poolpad")
if (Get-ChildItem "experiments\relgt\outputs\${pfx}relgt_chain_pool_${suf}_s*\holdout.npz" -ErrorAction SilentlyContinue) {
    $cfgs += "${pfx}relgt_chain_pool"
}
$cfgs += "${pfx}relgt_chain"
Write-Host "`n== own decisions: $($cfgs -join ', ')"
& $pyExe scripts\summarize_runs.py --suffix $suf @cfgs
Write-Host "`n== same decision rule on the same users: $($cfgs -join ', ')"
& $pyExe experiments\ensemble\run.py --suffix $suf @cfgs
