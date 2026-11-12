@echo off
setlocal
chcp 65001 >nul 2>&1
title Ocean Studio - Install Requirements

REM Run from this script's folder, even when launched from a shortcut.
cd /d "%~dp0"

echo ========================================================
echo       OCEAN STUDIO - REQUIREMENTS INSTALLER
echo ========================================================
echo.

if not exist "main.py" (
    echo [ERROR] main.py was not found next to this installer.
    echo Put this .bat and requirements.txt in your Ocean Studio folder.
    goto fail
)
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt was not found in this folder.
    echo Keep requirements.txt alongside this .bat file.
    goto fail
)

REM Use an existing project virtual environment, or create a new one.
if exist ".venv\Scripts\python.exe" (
    echo [INFO] Found .venv - reusing it.
    goto install
)

REM The existing application launcher prefers .venv over venv.
REM Reuse venv only if there is no .venv already.
if exist "venv\Scripts\python.exe" (
    echo [INFO] Found venv - reusing it.
    goto install_venv
)

echo [1/3] Creating the project Python environment...
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -m venv ".venv"
    if not errorlevel 1 goto install
)
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo Then run this installer again.
    goto fail
)
python -m venv ".venv"
if errorlevel 1 (
    echo [ERROR] Could not create a virtual environment.
    echo Make sure Python is installed with the venv module.
    goto fail
)

:install
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
goto install_requirements

:install_venv
set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"

:install_requirements
if not exist "%PYTHON_EXE%" (
    echo [ERROR] The environment's Python executable is missing.
    goto fail
)
echo.
echo [2/3] Updating pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Could not update pip. Check your internet connection.
    goto fail
)

echo.
echo [3/3] Installing Ocean Studio dependencies...
"%PYTHON_EXE%" -m pip install -r "requirements.txt"
if errorlevel 1 (
    echo [ERROR] One or more dependencies failed to install.
    echo Read the error above. Check internet access and free disk space.
    goto fail
)

echo.
echo [CHECK] Testing the required Python imports...
"%PYTHON_EXE%" -c "import cv2, numpy, PySide6, ultralytics; from ultralytics import YOLO; print('SUCCESS: All required libraries imported.')"
if errorlevel 1 (
    echo [ERROR] Libraries installed, but import verification failed.
    goto fail
)

echo.
echo ========================================================
echo [SUCCESS] Ocean Studio requirements are installed!
echo ========================================================
echo.

REM Ask only after all dependencies pass verification.
choice /C YN /N /M "Launch Ocean Studio now? [Y/N]: "
if errorlevel 2 goto finished
if errorlevel 1 goto launch_app

:launch_app
if exist "%~dp0Start_Ocean_Studio.bat" (
    echo.
    echo [INFO] Starting Ocean Studio...
    call "%~dp0Start_Ocean_Studio.bat"
) else (
    echo.
    echo [INFO] Start_Ocean_Studio.bat was not found.
    echo [INFO] Starting the application using the installed environment...
    "%PYTHON_EXE%" "main.py"
    if errorlevel 1 (
        echo [ERROR] Ocean Studio could not start. See the error above.
        pause
        exit /b 1
    )
)
goto finished

:finished
echo.
echo Setup finished. You can open Ocean Studio later using Start_Ocean_Studio.bat.
pause
exit /b 0

:fail
echo.
echo Installation did not complete. See the messages above.
pause
exit /b 1
