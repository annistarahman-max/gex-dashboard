@echo off
title Options Exposure Dashboard
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Starting Options Exposure Dashboard...
echo.
start http://localhost:8501
streamlit run app.py
pause
