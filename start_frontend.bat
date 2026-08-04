@echo off
setlocal
set "ROOT=%~dp0"
set "APP_DIR=%ROOT%frontend"

echo [FakeGuard] Starting frontend at http://localhost:5173 ...
start "FakeGuard Frontend" cmd /k "cd /d "%APP_DIR%" && npm run dev -- --port 5173"

endlocal
