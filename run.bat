@echo off

cd /d "%~dp0"

echo ================================================
echo   Plant Growth and Yield Prediction System
echo ================================================
echo.

echo Installing required packages...
py -3.10 -m pip install -r requirements.txt

echo.
echo Starting application...
echo.

py -3.10 main.py

pause