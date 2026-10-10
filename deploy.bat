@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Foley-Omni preview — one-shot deploy for Windows (Docker Desktop + NVIDIA).
REM Usage:
REM   deploy.bat
REM   deploy.bat --skip-download
REM   deploy.bat --force-download
REM   deploy.bat --no-logs

cd /d "%~dp0"

set "HF_REPO=CocoBro/Foley-Omni"
if defined HF_REPO_OVERRIDE set "HF_REPO=%HF_REPO_OVERRIDE%"

set "SKIP_DOWNLOAD=0"
set "FORCE_DOWNLOAD=0"
set "FOLLOW_LOGS=1"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--skip-download" (
  set "SKIP_DOWNLOAD=1"
  shift
  goto parse_args
)
if /I "%~1"=="--force-download" (
  set "FORCE_DOWNLOAD=1"
  shift
  goto parse_args
)
if /I "%~1"=="--no-logs" (
  set "FOLLOW_LOGS=0"
  shift
  goto parse_args
)
if /I "%~1"=="-h" goto help
if /I "%~1"=="--help" goto help
echo [deploy] Unknown argument: %~1
exit /b 1

:help
echo Foley-Omni one-shot deploy
echo.
echo   deploy.bat                   Download ckpts if needed, build and start Docker
echo   deploy.bat --skip-download   Skip Hugging Face download
echo   deploy.bat --force-download  Re-download checkpoints
echo   deploy.bat --no-logs         Start containers then exit
echo.
echo Env:
echo   HF_REPO_OVERRIDE   Hugging Face repo id override
exit /b 0

:args_done

echo [deploy] Project root: %CD%

where docker >nul 2>&1
if errorlevel 1 (
  echo [deploy] ERROR: docker not found in PATH.
  exit /b 1
)

set "PY="
where python >nul 2>&1
if not errorlevel 1 (
  set "PY=python"
) else (
  where py >nul 2>&1
  if not errorlevel 1 (
    set "PY=py -3"
  )
)
if not defined PY (
  echo [deploy] ERROR: Python not found. Install Python 3.10+ and retry.
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo [deploy] ERROR: Docker daemon is not reachable. Start Docker Desktop and retry.
  exit /b 1
)

set "COMPOSE=docker compose"
docker compose version >nul 2>&1
if errorlevel 1 (
  where docker-compose >nul 2>&1
  if errorlevel 1 (
    echo [deploy] ERROR: Docker Compose not found.
    exit /b 1
  )
  set "COMPOSE=docker-compose"
)

echo [deploy] Preparing .env tokens...
set "TOKEN_FILE=%TEMP%\foley_deploy_tokens.txt"
%PY% "%~dp0scripts\ensure_env_tokens.py" > "%TOKEN_FILE%"
if errorlevel 1 (
  echo [deploy] ERROR: failed to prepare .env tokens
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
  echo [deploy] ERROR: could not read ACCESS_TOKEN from ensure_env_tokens.py
  exit /b 1
)
if "%ACCESS_TOKEN%"=="%ADMIN_TOKEN%" (
  echo [deploy] ERROR: ACCESS_TOKEN and ADMIN_TOKEN must be different in .env
  exit /b 1
)

if /I "%TOKEN_STATUS%"=="GENERATED" (
  echo [deploy] Generated preview tokens ^(saved in .env^). Share ACCESS_TOKEN with guests only:
  echo          ACCESS_TOKEN=%ACCESS_TOKEN%
  echo          ADMIN_TOKEN=%ADMIN_TOKEN%
) else (
  echo [deploy] Using existing tokens from .env
)

if not exist "storage" mkdir storage
if not exist "ckpts" mkdir ckpts

if "%SKIP_DOWNLOAD%"=="0" (
  echo [deploy] Ensuring huggingface_hub is available...
  %PY% -m pip install -q -U "huggingface_hub[cli]>=0.30.0,<1.0"
  if errorlevel 1 (
    echo [deploy] ERROR: Failed to install huggingface_hub
    exit /b 1
  )
  echo [deploy] Downloading checkpoints from %HF_REPO% ...
  if "%FORCE_DOWNLOAD%"=="1" (
    %PY% "%~dp0scripts\download_release_ckpts.py" "%HF_REPO%" --root "%CD%" --force
  ) else (
    %PY% "%~dp0scripts\download_release_ckpts.py" "%HF_REPO%" --root "%CD%"
  )
  if errorlevel 1 (
    echo [deploy] ERROR: checkpoint download failed
    exit /b 1
  )
) else (
  echo [deploy] Skipping checkpoint download (--skip-download)
)

if exist "%~dp0scripts\check_setup.py" (
  echo [deploy] Verifying checkpoint layout...
  %PY% "%~dp0scripts\check_setup.py" --ckpt-dir "%CD%\ckpts" --model-checkpoint "%CD%\ckpts\Foley-Omni\v2st.pth"
  if errorlevel 1 (
    echo [deploy] ERROR: Checkpoint check failed. Re-run with --force-download or fix .\ckpts
    exit /b 1
  )
)

echo [deploy] Building and starting containers ^(first build can take a long time^)...
%COMPOSE% -f "%CD%\docker-compose.yml" --env-file "%CD%\.env" up --build -d
if errorlevel 1 (
  echo [deploy] ERROR: docker compose up failed
  exit /b 1
)

echo [deploy] Containers are up.
echo [deploy] UI:     http://localhost:8000/
echo [deploy] Health: http://localhost:8000/api/health
echo [deploy] Guest token ^(ACCESS_TOKEN^): %ACCESS_TOKEN%
echo [deploy] Useful: docker compose logs -f foley-omni
echo [deploy] Stop:   docker compose down

if "%FOLLOW_LOGS%"=="1" (
  echo [deploy] Tailing logs ^(Ctrl+C stops logs only; containers keep running^)...
  %COMPOSE% -f "%CD%\docker-compose.yml" logs -f foley-omni
)

exit /b 0
