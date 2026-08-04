@echo off
setlocal
echo [FakeGuard] Stopping backend (port 8000)...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do taskkill /F /T /PID %%a >nul 2>&1

rem uvicorn --reload spawns a separate worker process that can outlive the
rem PID netstat reports for the listening socket; sweep any leftover
rem interpreter still holding main:app open as a fallback.
for /f "skip=1 tokens=2" %%a in ('wmic process where "CommandLine like '%%main:app%%'" get ProcessId 2^>nul') do taskkill /F /PID %%a >nul 2>&1

echo Done.
endlocal
