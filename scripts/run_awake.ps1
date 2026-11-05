# Run a command while asking Windows not to sleep (released automatically when this process exits).
# Usage: powershell -File scripts/run_awake.ps1 bash scripts/run_extra.sh
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power]::SetThreadExecutionState([Convert]::ToUInt32("80000001", 16))  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED (a 0x literal parses as a negative Int32)
& $args[0] $args[1..($args.Length - 1)]
[void][Win.Power]::SetThreadExecutionState([Convert]::ToUInt32("80000000", 16))
