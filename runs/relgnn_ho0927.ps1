# RelGNN (experiments\relgnn: composite message passing over atomic routes) on the holdout week 2026-09-27 -> 10-04
# (Taiwan time; 09-26 16:00 UTC), all 6 activities. Same protocol as gnn_chain in runs\gnn_chain_bag_ho0927.ps1
# (experiments\gnn\run.py --holdout --holdout-final):
#   validation model: trained on weeks whose labels end by 09-19, early stopping (val loss, patience 5, at most
#                     -Epochs epochs) and the decision shares on the tune week 09-19 -> 09-26;
#   final model:      refit on every labelled week through 09-26 for the best epoch, scored with the same shares.
# Only the message passing over the tokens changes (--token-model relgnn):
#
#   config               graph   tokens                                           readout
#   relgnn_chain         chain   16 latest actions + BFS walk (gnn_chain's)       the user token (RelGNN's entity)
#   relgnn_chain_seeds   chain   the same                                         mean of the 16 seed actions (RowGNN's)
#   relgnn_fk            fk      the user + 99 rows of its 1- / 2-hop neighbours  the user token (the paper's graph)
#
# Tokens come from experiments\relgt\cache: the chain configs use gnn_chain's, relgnn_fk relgt_fk's (runs\
# sampling_ho0927.ps1; a seed without them is sampled once, ~1 min). One run: ~4.5 min per epoch on the chain
# graph, ~3 on fk (RTX 3060, 4.7 GiB at most), early stopping after best epoch + 5, then the final fit of best-epoch
# epochs: ~35 min if the best epoch is 1, ~70 min if 5. Jobs = configs x seeds, queued over -Gpus: each GPU runs one job at a time and
# takes the next when it is free (none of this script's jobs on it and less than -FreeMiB MiB in use, so a GPU
# another program is using is skipped until it frees up; runs\gpu.ps1). Runs are <cfg>_ens<MMDD>_s<seed> in
# experiments\gnn\outputs, logs in experiments\gnn\outputs\logs\<name>.log (new lines echoed every -EchoEvery s).
# Then scripts\summarize_runs.py (each run's own decisions) and experiments\ensemble\run.py with the references
# gnn_chain_ens0926_s0-2 (runs\gnn_chain_bag_ho0927.ps1) and relgt_chain_ens0926_s0-2 (runs\ensemble_ho0927.sh):
# seed bags under the same decision rule, final models, push by the tune-week share and by prev-week, and the mean
# of relgnn_chain with relgt_chain.
#   .\runs\relgnn_ho0927.ps1                                   # 3 configs x seeds 0-2 on GPUs 0 and 1
#   .\runs\relgnn_ho0927.ps1 -Configs relgnn_chain -Gpus 1     # one config, one GPU
#   .\runs\relgnn_ho0927.ps1 -Smoke                            # pipeline check: seed 0, 1 epoch of 30 steps
#   .\runs\relgnn_ho0927.ps1 -SkipRuns                         # only the summaries
param(
    [int[]]$Gpus = @(0, 1),
    [int[]]$Seeds = @(0, 1, 2),
    [string[]]$Configs = @("relgnn_chain", "relgnn_chain_seeds", "relgnn_fk"),
    [int]$Epochs = 20,               # validation stage: at most this many, early stopping on the tune week
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
$pfx = ""; $epochArgs = @("--epochs", "$Epochs")
if ($Smoke) { $pfx = "smoke_"; $Seeds = @(0); $epochArgs = @("--epochs", "1", "--max-steps-per-epoch", "30") }
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

$chain = @("--graph-structure", "chain", "--chain-seed", "latest", "--num-seeds", "16", "--token-model", "relgnn")
$cfg = [ordered]@{
    "relgnn_chain"       = $chain
    "relgnn_chain_seeds" = $chain + @("--relgnn-readout", "seeds")
    "relgnn_fk"          = @("--graph-structure", "fk", "--token-model", "relgnn")
}
$common = @("experiments\gnn\run.py", "--holdout", "--holdout-week", $HoWeek, "--holdout-final") + $epochArgs

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
        $score = Select-String -Path $run.Log -Encoding UTF8 -Pattern '^HOLDOUT_FINAL .*score' | Select-Object -Last 1
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
        if (-not $cfg.Contains($c)) { throw "unknown config $c (known: $($cfg.Keys -join ', '))" }
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

# The RelGNN configs that have runs, then the references (gnn_chain, relgt_chain) when theirs exist.
function Has-Runs([string]$c) {
    @(Get-ChildItem "experiments\gnn\outputs\${c}_${suf}_s*\holdout.npz", "experiments\relgt\outputs\${c}_${suf}_s*\holdout.npz" `
        -ErrorAction SilentlyContinue).Count -gt 0
}
$mine = @($cfg.Keys | ForEach-Object { "${pfx}$_" } | Where-Object { Has-Runs $_ })
$refs = @(@("gnn_chain", "relgt_chain") | ForEach-Object { "${pfx}$_" } | Where-Object { Has-Runs $_ })
if (-not $mine) { Write-Host "no RelGNN runs with suffix $suf"; exit 1 }
$all = $mine + $refs
Write-Host "`n== own decisions (mean over seeds; GNN runs: validation model, then final model): $($all -join ', ')"
& $pyExe scripts\summarize_runs.py --suffix $suf @all

function Ens([string[]]$a) {
    Write-Host "`n== ensemble: $($a -join ' ')"
    & $pyExe experiments\ensemble\run.py --suffix $suf @a
}
Ens ($all + @("--gnn-variant", "final"))                                  # every bag, the same decision rule
Ens ($all + @("--gnn-variant", "final", "--push-rule", "prev-week"))      # the submission's push rule
$rc = "${pfx}relgt_chain"; $nc = "${pfx}relgnn_chain"
if (($mine -contains $nc) -and ($refs -contains $rc)) {
    Ens @($nc, $rc, "--gnn-variant", "final")                             # mean of the two chain models
}
