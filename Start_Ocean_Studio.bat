@echo off
setlocal
title Ocean Studio - Underwater Vision AI

REM Always run from the folder containing this .bat file.
cd /d "%~dp0"

if not exist "main.py" (
    echo [ERROR] main.py was not found in this folder.
    echo Put Start_Ocean_Studio.bat next to main.py and try again.
    pause
    exit /b 1
)

REM Prefer a project virtual environment if one exists.
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
    goto run_app
)
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
    goto run_app
)

REM Otherwise use installed Python. Try the Windows Python launcher first.
where py >nul 2>&1
if not errorlevel 1 (
    py -3 "main.py"
    goto check_result
)
where python >nul 2>&1
if not errorlevel 1 (
    python "main.py"
    goto check_result
)

echo [ERROR] Python was not found.
echo Install Python or create a virtual environment and try again.
pause
exit /b 1

:run_app
"%PYTHON_EXE%" "main.py"

:check_result
if errorlevel 1 (
    echo.
    echo [ERROR] Ocean Studio stopped with an error.
    echo Read the message above to see what went wrong.
    pause
    exit /b 1
)
endlocal
