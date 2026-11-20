# Seed bagging of RelGT on a second holdout week: 2026-10-04 -> 10-11 (Taiwan time; 10-03 16:00 UTC), all 6
# activities. The first week (09-27 -> 10-04, runs/ensemble_ho0927.sh) gave bag:relgt_chain 54.42 against 52.59 for
# the single final-model runs, from 3 seeds on one week (docs/experiments/ensemble.md); this week checks it.
# Same protocol and training settings as runs/related_ho0927.ps1, shifted one week: 15 epochs; train on weeks whose
# labels end by 09-26, tune on 09-26 -> 10-03 (early stopping and the share of predicted positives), score the holdout
# week 10-03 -> 10-10 UTC; the validation model and the final model refit through 10-03.
#
#   config              sampling
#   relgt_chain         chain, 16 latest actions, fk-order random
#   relgt_chain_recent  chain, 16 latest actions, fk-order recent
#
# Week check (2026-11-19, read-only, eligible users and positive rate of eligible users):
#   week 09-26: 6,064 users, push 0.418     week 10-03: 5,731 users, push 0.355     (this one)
#   week 10-10: 5,443 users, push 0.359     week 10-17: 5,165 users, push 0.327
# The cohort's pushes are missing on 10-07 and about a third of normal on 10-08 -> 10-10, so the push labels of this week
# are understated; the other candidates lose 10-15 -> 10-18 or 10-23 as well. Non-push positive rates are 0.026-0.080.
#
# Pre-registered test (docs/experiments/relgt_bag_second_week.md): bag:<cfg> beats the mean of its three seed:<run>
# scores by more than 0.5 points, for both configs.
#
# Jobs = configs x seeds, queued over -Gpus: each GPU runs one job at a time and takes the next when it is free:
# none of this script's jobs on it and less than -FreeMiB (2000) MiB in use, so a GPU that another program is
# using is skipped until it frees up (runs\gpu.ps1).
# Every config samples its tokens once per seed (a few minutes; cached in experiments\relgt\cache), then
# trains ~15-25 min. Each job's output goes to experiments\gnn\outputs\logs\<name>.log; new lines of every
# running job are echoed every -EchoEvery seconds. Ctrl+C stops the running jobs too.
# Runs are named <cfg>_ens1003_s<seed>, so the 09-26 runs (ens0926) stay as they are.
#   .\runs\relgt_bag_ho1003.ps1                                  # 2 configs x seeds 0-2 on GPUs 0 and 1
#   .\runs\relgt_bag_ho1003.ps1 -Gpus 1                          # all jobs one after another on GPU 1
#   .\runs\relgt_bag_ho1003.ps1 -Smoke                           # pipeline check: seed 0, 1 epoch of 30 steps
#   .\runs\relgt_bag_ho1003.ps1 -SkipRuns                        # only the summaries
param(
    [int[]]$Gpus = @(0, 1),
    [int[]]$Seeds = @(0, 1, 2),
    [string[]]$Configs = @("relgt_chain", "relgt_chain_recent"),
    [string]$HoWeek = "2026-10-03 16:00",   # UTC start of the scored week (Saturday 16:00)
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
$pfx = ""; $epochs = @("--epochs", "15")
if ($Smoke) { $pfx = "smoke_"; $Seeds = @(0); $epochs = @("--epochs", "1", "--max-steps-per-epoch", "30") }
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

$chain = @("--graph-structure", "chain", "--chain-seed", "latest", "--num-seeds", "16")
$recent = @("--fk-order", "recent")
$cfg = @{
    "relgt_chain"        = $chain
    "relgt_chain_recent" = $chain + $recent
}
$common = @("experiments\relgt\run.py", "--end", $hoEnd, "--holdout-weeks", "1", "--val-start", $tune,
            "--val-end", $HoWeek) + $epochs

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
    $runArgs = $common + $cfg[$job.Cfg] + @("--seed", "$($job.Seed)", "--name", $job.Name)
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
    foreach ($c in $Configs) {
        if (-not $cfg.ContainsKey($c)) { throw "unknown config $c (known: $($cfg.Keys -join ', '))" }
        foreach ($s in $Seeds) { $queue.Enqueue(@{ Cfg = $c; Seed = $s; Name = "${pfx}${c}_${suf}_s$s" }) }
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

# Every config that has runs.
$all = $Configs | ForEach-Object { "${pfx}$_" } |
    Where-Object { Get-ChildItem "experiments\relgt\outputs\${_}_${suf}_s*\holdout.npz" -ErrorAction SilentlyContinue }
Write-Host "`n== own decisions (mean over seeds): $($all -join ', ')"
& $pyExe scripts\summarize_runs.py --suffix $suf @all
Write-Host "`n== same decision rule, single seeds and seed bags: $($all -join ', ')"
& $pyExe experiments\ensemble\run.py --suffix $suf @all
