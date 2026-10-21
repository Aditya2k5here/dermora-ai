@echo off
REM ===========================================================================
REM  Dermora AI - Windows launcher
REM
REM  Reuses the virtual environment that already has torch + ultralytics
REM  (D:\yolov8\.venv by default) so nothing large has to be downloaded, and
REM  only adds the handful of packages the app itself needs.
REM
REM  Override with:  set DERMORA_PY=C:\path\to\python.exe  before running.
REM ===========================================================================
setlocal
cd /d "%~dp0"

if defined DERMORA_PY (
    set "PY=%DERMORA_PY%"
    goto :run
)

if exist "D:\yolov8\.venv\Scripts\python.exe" (
    set "PY=D:\yolov8\.venv\Scripts\python.exe"
    goto :run
)

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
    goto :run
)

set "PY=python"

:run
echo.
echo   Using interpreter: %PY%
echo   Checking dependencies...

"%PY%" -c "import fastapi, uvicorn, multipart, reportlab, cv2, ultralytics" 2>nul
if errorlevel 1 (
    echo   Installing missing packages ^(one time^)...
    "%PY%" -m pip install --quiet fastapi uvicorn python-multipart reportlab opencv-python
    "%PY%" -c "import ultralytics" 2>nul || "%PY%" -m pip install --quiet ultralytics
)

echo   Running self test...
"%PY%" selftest.py
if errorlevel 1 (
    echo.
    echo   Self test reported problems - see the output above.
    echo   Press any key to start the server anyway, or Ctrl+C to abort.
    pause >nul
)

echo.
echo   Starting Dermora AI at http://127.0.0.1:8000
start "" http://127.0.0.1:8000
"%PY%" app.py

endlocal
