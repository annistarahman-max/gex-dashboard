@echo off
title Options Exposure Dashboard
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Starting Options Exposure Dashboard...
echo.
start /b streamlit run app.py
timeout /t 3 /nobreak >nul
start http://localhost:8501
pause
