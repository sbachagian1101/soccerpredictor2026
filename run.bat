@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo          SOCCER PREDICTOR - WINDOWS
echo ==============================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set PYTHON=py
) else (
    set PYTHON=python
)

%PYTHON% -c "import pandas,numpy" >nul 2>nul
if errorlevel 1 (
    echo Installing required Python packages...
    %PYTHON% -m pip install -r requirements.txt
)

%PYTHON% desktop_app.py
if errorlevel 1 pause
endlocal
