@echo off
setlocal
REM Shows percent-complete for every training / evaluation / download job that
REM has written a log under backend\checkpoints\logo_runs.
REM
REM Pass "watch" as the first argument to refresh every 20 seconds:
REM     progress.bat watch
REM
REM Note: Python block-buffers stdout when it is redirected to a file, so a
REM job can be further along than its log shows. Checkpoint files are written
REM unbuffered, so the training rows are the most trustworthy.

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%backend\training\progress_status.py"
set "LOGS=%ROOT%backend\checkpoints\logo_runs"

if not exist "%PY%" (
    echo ERROR: no virtualenv python at "%PY%"
    echo Expected: "%PY%"
    echo Run this from the project root, with the .venv present.
    pause
    exit /b 1
)
if not exist "%LOGS%" (
    echo No job logs yet - nothing has been run.
    echo Expected directory: "%LOGS%"
    pause
    exit /b 0
)

if /i "%~1"=="watch" goto watch

"%PY%" "%SCRIPT%" --scan-dir "%LOGS%"
if errorlevel 1 (
    echo.
    echo progress_status.py exited with an error - see above.
)
pause
goto end

:watch
echo Refreshing every 20s. Press Ctrl+C to stop.
:loop
cls
echo FakeGuard job progress   %DATE% %TIME%
echo.
"%PY%" "%SCRIPT%" --scan-dir "%LOGS%"
timeout /t 20 /nobreak >nul
goto loop

:end
endlocal
