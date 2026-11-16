# One-step submission with the best model: seed bag of RelGT on the chain graph with --fk-order recent
# (relgt_chain_recent). Trains one final model per seed on every labelled week (no holdout), bags the seeds
# (experiments\relgt\bag_submit.py) and checks the format with the course's evaluate.py.
#
#   .\run_submission.ps1                          # seeds 0-2 on GPU 0 (one after the other)
#   .\run_submission.ps1 -Gpus 0,1                # seeds spread over two GPUs, one job per GPU at a time
#   .\run_submission.ps1 -Smoke                   # pipeline check: seed 0, 1 epoch of 30 steps
#   .\run_submission.ps1 -SkipTrain               # only bag and check runs that already exist
#   .\run_submission.ps1 -StudentId 123456        # also copies the result to output_123456.csv in the repo root
#
# Output: experiments\relgt\outputs\relgt_chain_recent_bag\output.csv (one row per test user, six 0/1 columns).
# Each seed takes about 30 min (token sampling ~6.5 min, cached in experiments\relgt\cache). Logs of the runs:
# experiments\relgt\outputs\logs\<run>.log. Ctrl+C stops the running jobs.
param(
    [int[]]$Gpus = @(0),
    [int[]]$Seeds = @(0, 1, 2),
    [string]$Name = "relgt_chain_recent",   # run folders are <Name>_test_s<seed>, the bag <Name>_bag
    [string]$StudentId = "",
    [string]$Python = "",
    [switch]$Smoke,
    [switch]$SkipTrain
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:TQDM_MININTERVAL = "10"

if (-not $Python) {
    if ($env:VIRTUAL_ENV) { $Python = "python" }
    elseif (Test-Path ".venv-win\Scripts\python.exe") { $Python = ".venv-win\Scripts\python.exe" }
    else { $Python = "python" }
}
$py = (Get-Command $Python).Source
$pfx = ""; $fast = @("--epochs", "15")
if ($Smoke) { $pfx = "smoke_"; $Seeds = @(0); $fast = @("--epochs", "1", "--max-steps-per-epoch", "30") }
$logDir = "experiments\relgt\outputs\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$runs = $Seeds | ForEach-Object { "${pfx}${Name}_test_s$_" }
$bag = "${pfx}${Name}_bag"

function Now { Get-Date -Format "yyyy-MM-dd HH:mm:ss" }

if (-not $SkipTrain) {
    $base = @("experiments\relgt\run.py", "--graph-structure", "chain", "--chain-seed", "latest", "--num-seeds", "16",
              "--fk-order", "recent", "--holdout-weeks", "0") + $fast
    $queue = New-Object System.Collections.Queue
    foreach ($s in $Seeds) { $queue.Enqueue(@{ Seed = $s; Name = "${pfx}${Name}_test_s$s" }) }
    $running = @{}                       # gpu -> @{ Proc; Name; Log }
    $failed = 0
    try {
        while ($queue.Count -gt 0 -or $running.Count -gt 0) {
            foreach ($g in $Gpus) {
                if ($running.ContainsKey($g) -and $running[$g].Proc.HasExited) {
                    $r = $running[$g]
                    if ($r.Proc.ExitCode -eq 0) { Write-Host "$(Now) [gpu$g $($r.Name)] DONE" }
                    else { Write-Host "$(Now) [gpu$g $($r.Name)] FAILED (exit $($r.Proc.ExitCode)), see $($r.Log)"; $failed++ }
                    $running.Remove($g)
                }
                if (-not $running.ContainsKey($g) -and $queue.Count -gt 0) {
                    $job = $queue.Dequeue()
                    $log = Join-Path $logDir "$($job.Name).log"
                    $runArgs = $base + @("--seed", "$($job.Seed)", "--name", $job.Name)
                    $env:CUDA_VISIBLE_DEVICES = "$g"
                    $p = Start-Process -FilePath $py -ArgumentList $runArgs -WorkingDirectory $PSScriptRoot `
                        -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -PassThru
                    $null = $p.Handle
                    Write-Host "$(Now) [gpu$g $($job.Name)] START (pid $($p.Id)) -> $log"
                    $running[$g] = @{ Proc = $p; Name = $job.Name; Log = $log }
                }
            }
            Start-Sleep -Seconds 15
        }
    } finally {
        foreach ($r in $running.Values) {
            if (-not $r.Proc.HasExited) { Stop-Process -Id $r.Proc.Id -Force -ErrorAction SilentlyContinue }
        }
    }
    if ($failed) { throw "$failed training run(s) failed; nothing was bagged" }
}

Write-Host "`n$(Now) bagging $($runs -join ', ')"
& $py experiments\relgt\bag_submit.py @runs --name $bag
if ($LASTEXITCODE) { throw "bag_submit.py failed" }
$out = "experiments\relgt\outputs\$bag\output.csv"

if (Test-Path "assignment2\scripts\evaluate.py") {
    Write-Host "`n$(Now) format check"
    & $py assignment2\scripts\evaluate.py --pred $out --input assignment2\submission\test_input.csv
}
if ($StudentId) {
    Copy-Item $out "output_$StudentId.csv" -Force
    Write-Host "copied to output_$StudentId.csv"
}
Write-Host "`nsubmission file: $out"
