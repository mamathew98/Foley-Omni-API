@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Foley-Omni — local UI smoke test (NO model / NO GPU weights).
REM Requires Python 3.10–3.13 (default "python" on PATH may be 3.14, which breaks pydantic).

cd /d "%~dp0"

echo [ui-local] Project root: %CD%

set "PY_CMD="
set "PY_VER="

REM Prefer the launcher with an explicit supported version (project targets 3.10).
for %%V in (3.10 3.11 3.12 3.13) do (
  if not defined PY_CMD (
    py -%%V -c "import sys" >nul 2>&1
    if not errorlevel 1 (
      set "PY_CMD=py -%%V"
      set "PY_VER=%%V"
    )
  )
)

if not defined PY_CMD (
  where python >nul 2>&1
  if not errorlevel 1 (
    for /f "delims=" %%A in ('python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"') do set "PY_VER=%%A"
    if "!PY_VER!"=="3.10" set "PY_CMD=python"
    if "!PY_VER!"=="3.11" set "PY_CMD=python"
    if "!PY_VER!"=="3.12" set "PY_CMD=python"
    if "!PY_VER!"=="3.13" set "PY_CMD=python"
  )
)

if not defined PY_CMD (
  echo [ui-local] ERROR: Need Python 3.10–3.13 for this UI smoke test.
  echo [ui-local] Your default interpreter is likely 3.14, which cannot install the pinned FastAPI/Pydantic stack.
  echo [ui-local] Install Python 3.10+ from https://www.python.org/downloads/ and ensure "py -3.10" works.
  exit /b 1
)

echo [ui-local] Using Python %PY_VER% via: %PY_CMD%

set "VENV_DIR=%CD%\.venv-ui"
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [ui-local] Creating virtualenv at .venv-ui ...
  %PY_CMD% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ui-local] ERROR: failed to create venv
    exit /b 1
  )
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
echo [ui-local] Installing lightweight web dependencies into .venv-ui ...
"%VENV_PY%" -m pip install -q --upgrade pip
"%VENV_PY%" -m pip install -q -r "%~dp0requirements-web.txt" omegaconf
if errorlevel 1 (
  echo [ui-local] ERROR: pip install failed
  exit /b 1
)

echo [ui-local] Preparing .env tokens...
set "TOKEN_FILE=%TEMP%\foley_ui_tokens.txt"
"%VENV_PY%" "%~dp0scripts\ensure_env_tokens.py" > "%TOKEN_FILE%"
if errorlevel 1 (
  echo [ui-local] ERROR: ensure_env_tokens.py failed
  exit /b 1
)

set "TOKEN_STATUS="
set "ACCESS_TOKEN="
set "ADMIN_TOKEN="
set "_n=0"
for /f "usebackq delims=" %%L in ("%TOKEN_FILE%") do (
  set /a _n+=1
  if !_n!==1 set "TOKEN_STATUS=%%L"
  if !_n!==2 set "ACCESS_TOKEN=%%L"
  if !_n!==3 set "ADMIN_TOKEN=%%L"
)
del "%TOKEN_FILE%" >nul 2>&1

if not defined ACCESS_TOKEN (
  echo [ui-local] ERROR: could not read tokens from .env
  exit /b 1
)

if not exist "storage" mkdir storage

set "SKIP_MODEL_LOAD=1"
set "ACCESS_TOKEN=%ACCESS_TOKEN%"
set "ADMIN_TOKEN=%ADMIN_TOKEN%"
set "STORAGE_DIR=%CD%\storage"
set "CKPT_DIR=%CD%\ckpts"

echo.
echo [ui-local] UI-only mode (SKIP_MODEL_LOAD=1^) — inference is disabled.
echo [ui-local] Open:  http://127.0.0.1:8000/
echo [ui-local] Guest token:  %ACCESS_TOKEN%
echo [ui-local] Admin token:  %ADMIN_TOKEN%
echo [ui-local] Press Ctrl+C to stop.
echo.

"%VENV_PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
exit /b %ERRORLEVEL%
