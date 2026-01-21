@echo off
REM MTBM Unified Calculator Launcher
REM ================================

cd /d "C:\Users\abdul\Desktop\ML for Tunneling"

echo.
echo ================================================
echo   MTBM Unified Calculator
echo   Microtunneling and Pipejacking Calculations
echo ================================================
echo.

REM Run the unified calculator CLI
python mtbm_calculator_cli.py %*

pause
