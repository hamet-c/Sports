# Registers Windows Scheduled Tasks for the NBA props pipeline.
#
# Tasks created (user-scope, no admin required):
#   - NbaProps_IngestProps     every 4h, runs ingest_props.py
#   - NbaProps_RefreshInjuries every 6h, runs refresh_injuries.py
#
# Each task logs to data/logs/<task>_<yyyymmdd>.log so you can audit runs.
# Re-run this script any time to reset triggers / paths.

$ErrorActionPreference = "Stop"

# Resolve absolute paths relative to this script.
$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir  = Resolve-Path (Join-Path $scriptDir "..")
$repoRoot    = Resolve-Path (Join-Path $backendDir "..")
$venvPython  = Join-Path $backendDir ".venv\Scripts\python.exe"
$logDir      = Join-Path $repoRoot "data\logs"

if (-not (Test-Path $venvPython)) {
    throw "venv python not found at $venvPython — create the venv first."
}
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Register-IngestTask {
    param(
        [Parameter(Mandatory)] [string] $TaskName,
        [Parameter(Mandatory)] [string] $ScriptName,
        [Parameter(Mandatory)] [int]    $IntervalHours,
        [Parameter(Mandatory)] [string] $StartTime  # "HH:mm"
    )

    $scriptPath = Join-Path $backendDir "scripts\$ScriptName"
    $logFile    = Join-Path $logDir   "$TaskName.log"

    # PowerShell wrapper so we get a single quotable command line, log capture,
    # and a deterministic working directory regardless of where Task Scheduler
    # decides to start us.
    $psCommand = @"
Set-Location -LiteralPath '$backendDir'
& '$venvPython' '$scriptPath' *>> '$logFile'
"@
    $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($psCommand))

    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded" `
        -WorkingDirectory $backendDir

    $trigger = New-ScheduledTaskTrigger -Daily -At $StartTime
    $trigger.Repetition = (New-ScheduledTaskTrigger `
        -Once -At $StartTime `
        -RepetitionInterval (New-TimeSpan -Hours $IntervalHours) `
        -RepetitionDuration (New-TimeSpan -Days 365)).Repetition

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
        -MultipleInstances IgnoreNew

    # Run as the current user, no elevation, only when logged on (so .env is
    # readable from your profile).
    $principal = New-ScheduledTaskPrincipal `
        -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType Interactive `
        -RunLevel Limited

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "NBA props pipeline: $ScriptName" `
        -Force | Out-Null

    Write-Host "Registered $TaskName  (every ${IntervalHours}h, start $StartTime, log: $logFile)"
}

Register-IngestTask `
    -TaskName "NbaProps_IngestProps" `
    -ScriptName "ingest_props.py" `
    -IntervalHours 4 `
    -StartTime "08:15"

Register-IngestTask `
    -TaskName "NbaProps_RefreshInjuries" `
    -ScriptName "refresh_injuries.py" `
    -IntervalHours 6 `
    -StartTime "08:00"

Write-Host ""
Write-Host "Done. Inspect with:"
Write-Host "  Get-ScheduledTask -TaskName 'NbaProps_*' | Get-ScheduledTaskInfo"
Write-Host "  Start-ScheduledTask -TaskName 'NbaProps_IngestProps'   # run now"
Write-Host "  Unregister-ScheduledTask -TaskName 'NbaProps_*' -Confirm:`$false   # remove"
