# RelGT with the baseline's 60 per-user features (--history), on the holdout week 2026-09-27 -> 10-04 (Taiwan time;
# 09-26 16:00 UTC), all 6 activities. Same protocol and training settings as runs/sampling_ho0927.ps1 (15 epochs; train on
# weeks whose labels end by 09-19, tune on 09-19 -> 09-26, score the holdout week; the validation model and the final
# model refit through 09-26). Only the user features change:
#
#   config                   sampling                                      user features
#   relgt_chain              chain, 16 latest actions, fk-order random     none          (reference, already run)
#   relgt_chain_recent       chain, 16 latest actions, fk-order recent     none          (reference, already run)
#   relgt_chain_hist         chain, 16 latest actions, fk-order random     --history     (reference, already run)
#   relgt_chain_recent_hist  chain, 16 latest actions, fk-order recent     --history
#
# --history: the baseline's per-user features (counts over 7 / 14 / 30 / 90 days and overall, days since the last event,
# active days, repositories, open PRs and issues; experiments/baseline/run.py build_week) are log1p-ed, standardised on
# the training examples, embedded by an MLP (Linear-ReLU-Linear) and added to every seed token, i.e. to each of the 16
# latest actions of the user (--chain-seed latest --num-seeds 16). The same idea gave the ego GNN about +2 points.
# The reference runs <cfg>_ens0926_s0-2 exist (runs/ensemble_ho0927.sh, runs/sampling_ho0927.ps1); the new config is run
# here as <cfg>_ens0926_s<seed>. Then scripts\summarize_runs.py (each run's own decisions) and
# experiments/ensemble/run.py over the four configs (single seeds, seed bags, the same decision rule, paired bootstrap).
#
# Jobs = configs x seeds, queued over -Gpus: each GPU runs one job at a time and takes the next when it is free:
# none of this script's jobs on it and less than -FreeMiB (2000) MiB in use, so a GPU that another program is
# using is skipped until it frees up (runs\gpu.ps1).
# Every new config samples its tokens once per seed (a few minutes; cached in experiments\relgt\cache), then
# trains ~15-25 min. Each job's output goes to experiments\gnn\outputs\logs\<name>.log; new lines of every
# running job are echoed every -EchoEvery seconds. Ctrl+C stops the running jobs too.
#   .\runs\relgt_hist_ho0927.ps1                                    # seeds 0-2 on GPUs 0 and 1
#   .\runs\relgt_hist_ho0927.ps1 -Seeds 0 -Gpus 1                   # one seed on one GPU
#   .\runs\relgt_hist_ho0927.ps1 -Smoke                             # pipeline check: seed 0, 1 epoch of 30 steps
#   .\runs\relgt_hist_ho0927.ps1 -SkipRuns                          # only the summaries
param(
    [int[]]$Gpus = @(0, 1),
    [int[]]$Seeds = @(0, 1, 2),
    [string[]]$Configs = @("relgt_chain_recent_hist"),
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
    "relgt_chain"                = $chain
    "relgt_chain_recent"         = $chain + $recent
    "relgt_chain_hist"           = $chain + @("--history")
    "relgt_chain_recent_hist"    = $chain + $recent + @("--history")
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

# Every config of the grid that has runs (the reference included).
$all = @("relgt_chain", "relgt_chain_recent", "relgt_chain_hist", "relgt_chain_recent_hist") | ForEach-Object { "${pfx}$_" } |
    Where-Object { Get-ChildItem "experiments\relgt\outputs\${_}_${suf}_s*\holdout.npz" -ErrorAction SilentlyContinue }
Write-Host "`n== own decisions (mean over seeds): $($all -join ', ')"
& $pyExe scripts\summarize_runs.py --suffix $suf @all
Write-Host "`n== same decision rule, seed bags: $($all -join ', ')"
& $pyExe experiments\ensemble\run.py --suffix $suf @all
