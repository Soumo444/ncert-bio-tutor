@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python 3.12 environment...
  uv venv --python 3.12 .venv
  uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
)

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo.
  echo Created .env - paste your free API keys into it:
  echo   Groq    https://console.groq.com/keys       ^(answers^)
  echo   Gemini  https://aistudio.google.com/apikey   ^(embeddings + judge^)
  echo.
  notepad ".env"
)

if not exist "data\manifest.json" (
  echo First run: indexing the documents in data\docs ...
  echo This takes a few minutes on the free embedding tier.
  ".venv\Scripts\python.exe" ingest.py
)

echo Starting RAG chatbot on http://127.0.0.1:8000
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep 3; Start-Process 'http://127.0.0.1:8000'"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
