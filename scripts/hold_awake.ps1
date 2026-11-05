# Keep Windows awake until a file contains a marker line (for jobs already running).
# Usage: powershell -File scripts/hold_awake.ps1 runs/queue_extra.log "EXTRA DONE"
param([string]$File, [string]$Marker)
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power]::SetThreadExecutionState([Convert]::ToUInt32("80000001", 16))
while (-not ((Test-Path $File) -and (Select-String -Path $File -Pattern $Marker -Quiet))) { Start-Sleep 30 }
[void][Win.Power]::SetThreadExecutionState([Convert]::ToUInt32("80000000", 16))
