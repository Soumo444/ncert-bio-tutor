@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Document AI - public link
echo ============================================================
echo.

if not exist "cloudflared.exe" (
  echo Downloading cloudflared once...
  powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'cloudflared.exe'"
  if errorlevel 1 (
    echo Download failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
)

powershell -NoProfile -Command "try { Invoke-RestMethod 'http://127.0.0.1:8000/api/health' -TimeoutSec 3 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
  echo Starting the chatbot server...
  start "Document AI server" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
  timeout /t 5 /nobreak >nul
) else (
  echo Server already running.
)

echo.
echo Creating a public https://...trycloudflare.com link.
echo Keep this window open while sharing. Press Ctrl+C to stop the link.
echo.
cloudflared.exe tunnel --url http://127.0.0.1:8000 --no-autoupdate

echo.
echo Tunnel stopped.
pause
