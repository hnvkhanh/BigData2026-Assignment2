# GPU availability for the PowerShell run scripts (the same test as pick_gpu in runs/lib.sh). Dot-source it:
#   . (Join-Path $PSScriptRoot "gpu.ps1")
# A GPU is free when its memory in use is under $GpuFreeMiB (default 2000 MiB; idle cards here use ~200-700 MiB),
# so a card that another program (or another run) is using is not given a new job. nvidia-smi does not always
# list the processes on a card, so memory in use is the test.

if (-not (Get-Variable GpuFreeMiB -Scope Script -ErrorAction SilentlyContinue)) { $script:GpuFreeMiB = 2000 }

function Get-GpuMemoryUsed {
    # hashtable: GPU index -> MiB in use
    $used = @{}
    foreach ($row in (& nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)) {
        $i, $m = $row -split ",\s*"
        $used[[int]$i] = [int]$m
    }
    if (-not $used.Count) { throw "nvidia-smi gave no GPU" }
    $used
}

function Test-GpuFree([int]$gpu) {
    $used = Get-GpuMemoryUsed
    if (-not $used.ContainsKey($gpu)) { throw "no GPU $gpu (nvidia-smi lists $(($used.Keys | Sort-Object) -join ', '))" }
    $used[$gpu] -lt $script:GpuFreeMiB
}

function Wait-FreeGpu([int[]]$candidates, [int]$waitSeconds = 60) {
    # the least used of the candidate GPUs (all GPUs if none given) once one is free; waits until then
    while ($true) {
        $used = Get-GpuMemoryUsed
        $pool = if ($candidates) { $candidates } else { @($used.Keys) }
        $best = $pool | Where-Object { $used.ContainsKey($_) } | Sort-Object { $used[$_] } | Select-Object -First 1
        if ($null -eq $best) { throw "none of GPUs $($pool -join ', ') exists" }
        if ($used[$best] -lt $script:GpuFreeMiB) {
            Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') gpu$best free ($($used[$best]) MiB in use)"
            return $best
        }
        Write-Host ("$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') no free GPU among $($pool -join ', ') (least used: " +
            "gpu$best, $($used[$best]) MiB >= $($script:GpuFreeMiB) MiB); retry in ${waitSeconds}s")
        Start-Sleep -Seconds $waitSeconds
    }
}
