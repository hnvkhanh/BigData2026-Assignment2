# Retrain the 10 best models of the 09-27 holdout (README, 3 seeds each) on data before June only.
# PowerShell, with a job queue over several GPUs.
#   relgt_chain 53.12 (final model) | ego_uf_qprev 53.08 | all_ego_uf_rel 53.06 | all_ego_uf 52.98
#   het 52.77 | gnn_chain 52.48 | het_notime 52.34 | user_repo 52.29 | ego_uf_qprev5 52.25
#   all_ego_uf_rel_pprev 52.19
# Same commands as script_ho0927.sh, script_chain_ho0927.sh and script_qprev_ho0927.sh, on another week.
# Holdout week 2026-05-23 16:00 UTC (05-24 -> 05-31 Taiwan time), tune week 05-16, so training labels
# end by 05-16 and features / tuning never see June or later (relgt_chain's "final model" is also
# retrained on the tune week, labels to 05-23). The week is fully labelled, so it is scored on all 6
# activities (push included). experiments\gnn\run.py jobs write experiments\gnn\outputs\<name>\,
# relgt_chain writes experiments\relgt\outputs\<name>\.
#
# Jobs = configs x seeds. Each GPU in -Gpus runs one job at a time; when a job ends, that GPU takes the
# next job of the queue, so every GPU stays busy until the queue is empty. The queue is in -Configs order:
# the slow graph models first, so the short ego jobs fill the gaps at the end. relgt_chain and gnn_chain
# sample the same token cache, so they are not next to each other (the cache write is safe anyway).
# Each job's output goes to experiments\gnn\outputs\logs\<name>.log.
# Progress: live PowerShell progress bars at the top of the console, one for the queue (jobs done /
# total) and one per GPU (its job's epoch, batch, % and time left, read from run.py's tqdm bars, which
# go to <name>.stderr.log while the job runs). Every -StatusEvery seconds the same is also printed as
# text, so it stays in the scrollback. Ctrl+C stops the running jobs too.
#
# Smoke test (default): -Epochs 2, batched jobs (ego, chain, relgt) capped at -Steps 30 batches per
# epoch; scores only check that the pipeline runs, they are not comparable to full runs.
# Full training: -Epochs 0 -Steps 0 -Prefix full  (0 epochs = the model's default: ego / chain 20,
# hetero / user-repo 40, with early stopping; relgt_chain 15 as in script_chain_ho0927.sh; 0 steps = full
# epochs)
#   .\script_smoke_may.ps1                                            # smoke test on GPUs 0 and 1
#   .\script_smoke_may.ps1 -Epochs 0 -Steps 0 -Prefix full            # full training on GPUs 0 and 1
#   .\script_smoke_may.ps1 -Epochs 0 -Steps 0 -Prefix full -Seeds 0,1,2
#   .\script_smoke_may.ps1 -Gpus 1 -Configs het                       # one GPU, one config
#   .\script_smoke_may.ps1 -Configs relgt_chain,ego_uf_qprev,all_ego_uf_rel    # a subset
param(
    [int[]]$Gpus = @(0, 1),
    [int[]]$Seeds = @(42),
    [int]$Epochs = 2,
    [int]$Steps = 30,
    [string[]]$Configs = @("relgt_chain", "het", "gnn_chain", "het_notime", "user_repo",
                           "all_ego_uf_rel", "ego_uf_qprev", "all_ego_uf", "ego_uf_qprev5",
                           "all_ego_uf_rel_pprev"),
    [string]$HoWeek = "2026-05-23 16:00",
    [string]$Python = "",
    [string]$Prefix = "smoke",   # output folders are <Prefix>_<cfg>_ho<MMDD>_s<seed>
    [int]$StatusEvery = 120,     # seconds between text status lines (the progress bars update every 5 s)
    [switch]$NoBars              # text status lines only, no progress bars
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$suf = "ho" + $HoWeek.Substring(5, 5).Replace("-", "")
# relgt takes the weeks as a data period: tune week = the week before, data end = a week after
$hoT = [datetime]::ParseExact($HoWeek, "yyyy-MM-dd HH:mm", [Globalization.CultureInfo]::InvariantCulture)
$tune = $hoT.AddDays(-7).ToString("yyyy-MM-dd HH:mm")
$hoEnd = $hoT.AddDays(7).ToString("yyyy-MM-dd HH:mm")
$logDir = Join-Path $PSScriptRoot "experiments\gnn\outputs\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"                  # UTF-8 output and open()
$env:TQDM_MININTERVAL = "5"            # tqdm bars update at most every 5 s (keeps the stderr file small)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Python: -Python, else the active venv, else .venv-win
if (-not $Python) {
    if ($env:VIRTUAL_ENV) { $Python = "python" }
    elseif (Test-Path ".venv-win\Scripts\python.exe") { $Python = ".venv-win\Scripts\python.exe" }
    else { $Python = "python" }
}
$pyExe = (Get-Command $Python).Source
Write-Host "python: $pyExe"

# Each config: the script, its arguments (without epochs / seed / name), and the epochs used for -Epochs 0
# (0 = the script's own default). Steps only matter for the batched models (ego, chain, relgt).
$gnnHo = @("experiments\gnn\run.py", "--holdout", "--holdout-week", $HoWeek)
$stepArgs = @("--max-steps-per-epoch", "$Steps")
$chain = @("--graph-structure", "chain", "--chain-seed", "latest", "--num-seeds", "16")
$cfg = @{
    "relgt_chain"          = @{ Out = "relgt"; Epochs = 15
                                Args = @("experiments\relgt\run.py") + $chain + @("--end", $hoEnd, "--holdout-weeks", "1",
                                         "--val-start", $tune, "--val-end", $HoWeek) + $stepArgs }
    "gnn_chain"            = @{ Out = "gnn"; Epochs = 0; Args = $gnnHo + $chain + $stepArgs }
    "het"                  = @{ Out = "gnn"; Epochs = 0; Args = $gnnHo + @("--graph-structure", "hetero", "--task", "all") }
    "het_notime"           = @{ Out = "gnn"; Epochs = 0
                                Args = $gnnHo + @("--graph-structure", "hetero", "--task", "all", "--het-time", "none") }
    "user_repo"            = @{ Out = "gnn"; Epochs = 0; Args = $gnnHo + @("--graph-structure", "user-repo") }
    "all_ego_uf_rel"       = @{ Out = "gnn"; Epochs = 0; Args = $gnnHo + @("--user-features", "--relative-features") + $stepArgs }
    "all_ego_uf"           = @{ Out = "gnn"; Epochs = 0; Args = $gnnHo + @("--user-features") + $stepArgs }
    "ego_uf_qprev"         = @{ Out = "gnn"; Epochs = 0
                                Args = $gnnHo + @("--user-features", "--quiet-rule", "prev-week") + $stepArgs }
    "ego_uf_qprev5"        = @{ Out = "gnn"; Epochs = 0
                                Args = $gnnHo + @("--user-features", "--quiet-rule", "prev-week", "--active-min-events", "5") + $stepArgs }
    "all_ego_uf_rel_pprev" = @{ Out = "gnn"; Epochs = 0
                                Args = $gnnHo + @("--user-features", "--relative-features", "--push-share", "prev-week") + $stepArgs }
}

function Now { Get-Date -Format "yyyy-MM-dd HH:mm:ss" }
function Quote([string]$a) { if ($a -match '[\s"]') { '"' + $a.Replace('"', '\"') + '"' } else { $a } }

# the queue: configs x seeds
$queue = New-Object System.Collections.Queue
foreach ($c in $Configs) {
    if (-not $cfg.ContainsKey($c)) { Write-Host "unknown config $c (known: $($cfg.Keys -join ', '))"; continue }
    $ep = if ($Epochs -gt 0) { $Epochs } else { $cfg[$c].Epochs }   # 0 = the script's default
    foreach ($s in $Seeds) { $queue.Enqueue(@{ Cfg = $c; Seed = $s; Epochs = $ep; Name = "${Prefix}_${c}_${suf}_s$s" }) }
}
$total = $queue.Count
Write-Host "$(Now) $total jobs on GPUs $($Gpus -join ', '): $(($queue.ToArray() | ForEach-Object { $_.Name }) -join ', ')"

function Start-Run($job, [int]$gpu) {
    $log = Join-Path $logDir "$($job.Name).log"
    $err = Join-Path $logDir "$($job.Name).stderr.log"
    $runArgs = $cfg[$job.Cfg].Args + @("--seed", "$($job.Seed)", "--name", $job.Name)
    if ($job.Epochs -gt 0) { $runArgs += @("--epochs", "$($job.Epochs)") }
    $env:CUDA_VISIBLE_DEVICES = "$gpu"   # inherited by the new process
    $p = Start-Process -FilePath $pyExe -ArgumentList (($runArgs | ForEach-Object { Quote $_ }) -join " ") `
        -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $log -RedirectStandardError $err `
        -NoNewWindow -PassThru
    $null = $p.Handle                    # keep a handle, or ExitCode stays empty after the process ends
    Write-Host "$(Now) [gpu$gpu $($job.Name)] START ($script:started/$total, pid $($p.Id)) -> $log"
    @{ Job = $job; Proc = $p; Log = $log; Err = $err; Start = Get-Date }
}

function Read-Tail([string]$path, [int]$bytes = 16384) {
    # the end of a file another process is still writing
    try {
        $fs = [System.IO.File]::Open($path, "Open", "Read", "ReadWrite, Delete")
        try {
            $start = [math]::Max(0, $fs.Length - $bytes)
            $null = $fs.Seek($start, "Begin")
            $buf = New-Object byte[] ($fs.Length - $start)
            $n = $fs.Read($buf, 0, $buf.Length)
            [System.Text.Encoding]::UTF8.GetString($buf, 0, $n)
        } finally { $fs.Close() }
    } catch { "" }
}

function Get-RunProgress($run) {
    # latest tqdm state of a job: (percent of the whole job 0-100 or -1, status text)
    # tqdm redraws with \r and moves between its nested bars with ESC[A: keep the plain bar texts
    $segs = ((Read-Tail $run.Err) -replace "\x1b\[[0-9;]*[A-Za-z]", "") -split "[\r\n]+" | Where-Object { $_.Trim() }
    $outer = $segs | Where-Object { $_ -match '^\s*epochs:' } | Select-Object -Last 1     # bar over epochs
    $inner = $segs | Where-Object { $_ -match '^\s*(epoch \d+/\d+|hetero graphs):' } | Select-Object -Last 1
    $lastOut = ((Read-Tail $run.Log 4096) -split "[\r\n]+" | Where-Object { $_.Trim() } | Select-Object -Last 1)
    $mins = [math]::Round(((Get-Date) - $run.Start).TotalMinutes, 1)
    $pct = -1; $parts = @()
    $done = -1; $n = 0; $epTime = ""; $cur = 0; $frac = 0; $batch = ""
    if ($outer -match '\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]') {
        $done, $n, $epTime = [int]$Matches[1], [int]$Matches[2], $Matches[3]
    }
    if ($inner -match '^\s*(.*?):\s*(\d+)%[^|]*\|[^|]*\|\s*(\d+/\d+)\s*\[([^\]]*)\]') {
        $what, $p, $cnt, $t = $Matches[1], [int]$Matches[2], $Matches[3], $Matches[4]
        if ($what -match '^epoch (\d+)/(\d+)$') {   # batch bar of epoch cur (ego): newer than the epoch bar
            $cur, $frac = [int]$Matches[1], ($p / 100)
            if ($n -eq 0) { $n = [int]$Matches[2] }
            $done = [math]::Max($done, $cur - 1)
            $batch = "batch $cnt $p% [$t]"
        } elseif ($done -lt 0) {                   # building the hetero graphs, before the first epoch
            $batch = "$what $p% $cnt [$t]"
        }
    }
    if ($n -gt 0 -and $done -ge 0) {
        if ($cur -ne $done + 1) { $frac = 0 }
        $parts += "epoch $([math]::Min($done + 1, $n))/$n" + $(if ($epTime) { " [$epTime]" } else { "" })
        $pct = [math]::Min(100, [int](100 * ($done + $frac) / $n))
    }
    if ($batch) { $parts += $batch }
    if ($pct -lt 0 -and $run.Job.Epochs -gt 0) {
        # no tqdm bars (relgt): its "  epoch N  train loss ..." lines; a first fit with validation,
        # then ("chosen: ...") the final fit up to the best epoch, each counted as half of the job
        $log = Read-Tail $run.Log 1048576
        $i = $log.IndexOf("`nchosen:")
        $first = if ($i -ge 0) { $log.Substring(0, $i) } else { $log }
        $n1 = ([regex]::Matches($first, '(?m)^\s*epoch\s+\d+\s')).Count
        if ($i -ge 0) {
            $n2 = ([regex]::Matches($log.Substring($i), '(?m)^\s*epoch\s+\d+\s')).Count
            $b = if ($log -match 'best epoch (\d+)') { [int]$Matches[1] } else { $run.Job.Epochs }
            $parts = @("final fit epoch $([math]::Min($n2 + 1, $b))/$b")
            $pct = [math]::Min(100, [int](50 + 50 * $n2 / [math]::Max(1, $b)))
        } elseif ($n1 -gt 0 -or $log -match 'model parameters') {
            $parts = @("fit epoch $([math]::Min($n1 + 1, $run.Job.Epochs))/$($run.Job.Epochs)")
            $pct = [math]::Min(50, [int](50 * $n1 / $run.Job.Epochs))
        }
    }
    if (-not $parts) { $parts += "starting" }
    $text = "$($parts -join ' | ') | $mins min"
    if ($lastOut) { $text += " | $($lastOut.Trim())" }
    @($pct, $text)
}

function Complete-Run($run, [int]$gpu) {
    $run.Proc.WaitForExit()
    $rc = $run.Proc.ExitCode
    # stderr goes at the end of the log: the warnings / traceback, without the tqdm bar redraws
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
    if (-not $NoBars) { Write-Progress -Id (10 + $gpu) -ParentId 1 -Activity "gpu$gpu" -Completed }
    $mins = [math]::Round(((Get-Date) - $run.Start).TotalMinutes, 1)
    if ($rc -eq 0) {
        $score = Select-String -Path $run.Log -Encoding UTF8 -Pattern '^HOLDOUT .*users|^mean holdout score' |
            Select-Object -Last 1
        Write-Host "$(Now) [gpu$gpu $($run.Job.Name)] OK after $mins min  $($score.Line)"
    } else {
        Write-Host "$(Now) [gpu$gpu $($run.Job.Name)] FAILED (exit $rc) after $mins min, see $($run.Log)"
        Get-Content $run.Log -Tail 5 -Encoding UTF8 | ForEach-Object { Write-Host "    $_" }
    }
}

$running = @{}                           # gpu -> run
$script:started = 0
$finished = 0
$t0 = Get-Date
$lastStatus = Get-Date
try {
    while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        foreach ($g in $Gpus) {          # a finished job frees its GPU ...
            if ($running.ContainsKey($g) -and $running[$g].Proc.HasExited) {
                Complete-Run $running[$g] $g
                $running.Remove($g)
                $finished++
            }
            if (-not $running.ContainsKey($g) -and $queue.Count -gt 0) {   # ... which takes the next job
                $script:started++
                $running[$g] = Start-Run $queue.Dequeue() $g
            }
        }
        $prog = @{}
        foreach ($g in ($running.Keys | Sort-Object)) { $prog[$g] = Get-RunProgress $running[$g] }
        $elapsed = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        $overall = "$finished/$total done, $($running.Count) running, $($queue.Count) waiting | $elapsed min"
        if (-not $NoBars) {
            # the queue bar counts each running job by its own progress
            $partial = ($prog.Values | ForEach-Object { [math]::Max(0, $_[0]) / 100 } | Measure-Object -Sum).Sum
            Write-Progress -Id 1 -Activity "queue" -Status $overall `
                -PercentComplete ([math]::Min(100, [int](100 * ($finished + $partial) / [math]::Max(1, $total))))
            foreach ($g in $prog.Keys) {
                $pct, $text = $prog[$g]
                $bar = @{ Id = 10 + $g; ParentId = 1; Activity = "gpu$g $($running[$g].Job.Name)"; Status = $text }
                if ($pct -ge 0) { $bar.PercentComplete = $pct }
                Write-Progress @bar
            }
        }
        if (((Get-Date) - $lastStatus).TotalSeconds -ge $StatusEvery -and $running.Count -gt 0) {
            $lastStatus = Get-Date
            foreach ($g in ($prog.Keys | Sort-Object)) {
                Write-Host "$(Now) [gpu$g $($running[$g].Job.Name)] $($prog[$g][1])"
            }
            Write-Host "$(Now) queue: $overall"
        }
        Start-Sleep -Seconds 5
    }
} finally {                              # Ctrl+C or an error: do not leave python processes on the GPUs
    if (-not $NoBars) { Write-Progress -Id 1 -Activity "queue" -Completed }
    foreach ($r in $running.Values) {
        if (-not $r.Proc.HasExited) {
            Write-Host "$(Now) stopping $($r.Job.Name) (pid $($r.Proc.Id))"
            Stop-Process -Id $r.Proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
Write-Host "$(Now) all $total jobs finished"

# summary of every <Prefix> run of this week, best score first (the code goes in on stdin: no quoting
# problems). relgt_chain gives two rows: its final model and its validation model.
$summary = @"
import json, glob, os
rows = []
for f in glob.glob('experiments/gnn/outputs/${Prefix}_*_${suf}_s*/metrics.json'):
    m = json.load(open(f)); h = m['holdout']
    rows.append((os.path.basename(os.path.dirname(f)), h['score'], m.get('best_epoch'), h['f1']))
for f in glob.glob('experiments/relgt/outputs/${Prefix}_*_${suf}_s*/metrics.json'):
    m = json.load(open(f))
    for v in ('final model', 'validation model'):
        h = m['holdout'][0][v]
        rows.append((os.path.basename(os.path.dirname(f)) + ' (' + v.split()[0] + ')', h['score'], m.get('best_epoch'), h['per_event']))
for name, s, ep, f1 in sorted(rows, key=lambda r: -r[1]):
    print(f"{name:44s} score {100*s:.2f}  best_ep {ep}  " + '  '.join(f'{k} {v:.3f}' for k, v in f1.items()))
"@
$summary | & $pyExe -
