# Experiment: one RelGT per label (6 models instead of one 6-output model), on the holdout week 2026-09-27 -> 10-04
# (Taiwan time; 09-26 16:00 UTC), all 6 activities. Same protocol and training settings as relgt_chain_recent
# (runs/sampling_ho0927.ps1): 15 epochs; train on weeks whose labels end by 09-19, tune on 09-19 -> 09-26, score the
# holdout week; the validation model and the final model refit through 09-26.
#
# run.py --labels <label>: only that label is in the loss, the epoch choice (mean AP), the F1 threshold and the output;
# the other 5 outputs of the head get no gradient. No joint thresholds (they need all 6 labels). The 6 runs of a seed
# are named <cfg>_only_<label>_ens0926_s<seed>; experiments\relgt\merge_labels.py joins column j of each into one run,
# <cfg>_perlabel_ens0926_s<seed>, which is compared with the 6-output <cfg> (runs/sampling_ho0927.ps1, not rerun).
#
#   config               graph                              compared with (existing runs)
#   relgt_chain_recent   chain, 16 latest seeds, recent fill   relgt_chain_recent (one model, 6 outputs)
#
# Then scripts\summarize_runs.py and experiments\ensemble\run.py over <cfg> and <cfg>_perlabel (single seeds, seed bags,
# the same decision rule, paired bootstrap).
#
# Jobs = configs x labels x seeds (default 1 x 6 x seed 42 = 6 jobs), queued over -Gpus: each GPU runs one job at a
# time and takes the next when it is free:
# none of this script's jobs on it and less than -FreeMiB (2000) MiB in use, so a GPU that another program is
# using is skipped until it frees up (runs\gpu.ps1).
# The tokens do not depend on the label, so the 6 runs of a seed share one token cache (experiments\relgt\cache; the
# first run samples, a few minutes, the others reuse it), then each trains ~15-25 min. Each job's output goes to
# experiments\gnn\outputs\logs\<name>.log.
#   .\runs\relgt_label_ho0927.ps1                         # 6 labels, seed 42, on GPUs 0 and 1
#   .\runs\relgt_label_ho0927.ps1 -Seeds 42,43            # two seeds
#   .\runs\relgt_label_ho0927.ps1 -Labels push,open_pr -Gpus 1   # only these labels (merge needs all 6)
#   .\runs\relgt_label_ho0927.ps1 -Smoke                  # pipeline check: seed 42, 1 epoch of 30 steps
#   .\runs\relgt_label_ho0927.ps1 -SkipRuns               # only the merge and the summaries
param(
    [int[]]$Gpus = @(0, 1),
    [int[]]$Seeds = @(42),
    [string[]]$Labels = @("open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"),
    [string[]]$Configs = @("relgt_chain_recent"),
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
if ($Smoke) { $pfx = "smoke_"; $Seeds = @(42); $epochs = @("--epochs", "1", "--max-steps-per-epoch", "30") }
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
    "relgt_chain_recent"      = $chain + $recent
    "relgt_chain"             = $chain
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
    $runArgs = $common + $cfg[$job.Cfg] + @("--labels", $job.Label, "--seed", "$($job.Seed)", "--name", $job.Name)
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
        foreach ($s in $Seeds) {
            foreach ($l in $Labels) {
                $queue.Enqueue(@{ Cfg = $c; Seed = $s; Label = $l; Name = "${pfx}${c}_only_${l}_${suf}_s$s" })
            }
        }
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

# Merge the 6 single-label runs of each config and seed (all 6 labels must have run), then compare with the
# 6-output config.
foreach ($c in $Configs) {
    & $pyExe experiments\relgt\merge_labels.py --cfg $c --suffix $suf --seeds @Seeds "--prefix=$pfx"
    if ($LASTEXITCODE -ne 0) { Write-Host "merge of $c failed (are all 6 labels run?)" }
}
$all = $Configs | ForEach-Object { "${pfx}$_"; "${pfx}${_}_perlabel" } |
    Where-Object { Get-ChildItem "experiments\relgt\outputs\${_}_${suf}_s*\holdout.npz" -ErrorAction SilentlyContinue }
Write-Host "`n== own decisions (mean over seeds): $($all -join ', ')"
& $pyExe scripts\summarize_runs.py --suffix $suf @all
Write-Host "`n== same decision rule, seed bags: $($all -join ', ')"
& $pyExe experiments\ensemble\run.py --suffix $suf @all
