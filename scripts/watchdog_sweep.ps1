# Watchdog for the multi-seed sweep (scripts/run_multiseed_sweep.py).
#
# Run manually, or (recommended) registered as a repeating Windows Scheduled
# Task so it survives reboots without this machine needing a logged-in user
# or an open terminal/Claude Code session:
#   - If the sweep is already running (tracked via sweep.pid), does nothing.
#   - If the sweep is fully done (0 cells remaining), does nothing.
#   - Otherwise (crashed, machine rebooted, never started, etc.), launches it
#     again. run_multiseed_sweep.py is itself resumable -- a cell counts as
#     done iff its METRICS_ROW.json exists -- so relaunching never re-does
#     finished work.
#
# Manual one-off restart, no watchdog needed:
#   cd "C:\Users\Dr. Zarepour\Desktop\RL_IDS\Codes"
#   python scripts\run_multiseed_sweep.py

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$MultiseedDir = Join-Path $Root "experiments\MULTISEED"
New-Item -ItemType Directory -Force -Path $MultiseedDir | Out-Null
$WatchdogLog = Join-Path $MultiseedDir "watchdog.log"
$LockFile = Join-Path $MultiseedDir "sweep.pid"

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $WatchdogLog -Value $line
    Write-Output $line
}

# --- Is a sweep process already alive? ---
$running = $false
if (Test-Path $LockFile) {
    $trackedPid = Get-Content $LockFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($trackedPid) {
        $proc = Get-Process -Id $trackedPid -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -like "python*") {
            $running = $true
        }
    }
}
if ($running) {
    Log "Sweep already running (pid $trackedPid). Nothing to do."
    exit 0
}

# --- Is the sweep fully done? Ask the sweep script itself (--dry-run), so
#     this never duplicates the "what counts as done" logic. ---
$planOutput = & python scripts\run_multiseed_sweep.py --dry-run 2>&1 | Out-String
if ($planOutput -match "(\d+) cells total, (\d+) remaining") {
    $remaining = [int]$Matches[2]
} else {
    Log "Could not parse sweep plan output; assuming work remains. Output was:`n$planOutput"
    $remaining = -1
}

if ($remaining -eq 0) {
    Log "Sweep complete (0 cells remaining). Nothing to do."
    exit 0
}

Log "Sweep not running; $remaining cell(s) remaining. Launching run_multiseed_sweep.py ..."
$proc = Start-Process -FilePath "python" `
    -ArgumentList "scripts\run_multiseed_sweep.py" `
    -WorkingDirectory $Root `
    -RedirectStandardOutput (Join-Path $MultiseedDir "sweep_stdout.log") `
    -RedirectStandardError (Join-Path $MultiseedDir "sweep_stderr.log") `
    -WindowStyle Hidden -PassThru

$proc.Id | Out-File -FilePath $LockFile -Encoding ascii
Log "Launched sweep, pid=$($proc.Id). stdout/stderr -> $MultiseedDir\sweep_stdout.log / sweep_stderr.log"
