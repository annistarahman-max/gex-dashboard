@echo off
title GEX Dashboard Bot
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Starting GEX Dashboard Bot...
echo.
python bot.py
pause
