@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (set PYTHON=py) else (set PYTHON=python)
%PYTHON% -m pip install -r requirements.txt
%PYTHON% -m streamlit run streamlit_app.py
pause
endlocal
