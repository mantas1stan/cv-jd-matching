@echo off
setlocal

cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "APP_FILE=app\streamlit_app.py"
set "PORT=8501"

echo.
echo CV-JD Matching Research Demo
echo Project folder: %CD%
echo.

if not exist "%APP_FILE%" (
    echo ERROR: Could not find %APP_FILE%.
    echo Run this file from the uni_cv_jd project folder.
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo Creating local Python environment in .venv ...
    py -3 -m venv "%VENV_DIR%" 2>nul
    if errorlevel 1 (
        python -m venv "%VENV_DIR%"
    )
)

if not exist "%PYTHON_EXE%" (
    echo ERROR: Could not create .venv. Make sure Python is installed and available as py or python.
    pause
    exit /b 1
)

echo Checking Streamlit dependencies ...
"%PYTHON_EXE%" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Installing project requirements ...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
)

echo.
echo Starting Streamlit at http://localhost:%PORT%
echo Press Ctrl+C in this window to stop the app.
echo.

"%PYTHON_EXE%" -m streamlit run "%APP_FILE%" --server.port %PORT% --server.address localhost

echo.
echo Streamlit stopped.
pause
