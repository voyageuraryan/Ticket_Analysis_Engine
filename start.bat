@echo off
REM SAP Module Classifier - Windows Startup Script
REM This script launches the Streamlit application

echo ================================================================================
echo SAP MODULE CLASSIFIER - STARTUP
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python installation...
python --version
echo.

REM Check if virtual environment exists
if exist venv (
    echo [2/4] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [2/4] No virtual environment found. Using system Python...
)
echo.

REM Check if dependencies are installed
echo [3/4] Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Dependencies not installed. Installing now...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo Dependencies OK
)
echo.

REM Check if .env file exists
if not exist .env (
    echo WARNING: .env file not found!
    echo Please create .env file with your API keys:
    echo   GOOGLE_API_KEY=your_gemini_key
    echo   GROQ_API_KEY=your_groq_key
    echo.
    echo Press any key to continue anyway, or Ctrl+C to exit...
    pause >nul
)

REM Display menu
echo [4/4] Select application to launch:
echo.
echo   1. Original App (Classification only)
echo   2. Enhanced App (Classification + Employee Management + Assignment)
echo   3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Launching Original Streamlit App...
    echo Access the app at: http://localhost:8501
    echo Press Ctrl+C to stop the server
    echo.
    streamlit run app/streamlit_app.py
) else if "%choice%"=="2" (
    echo.
    echo Launching Enhanced Streamlit App...
    echo Access the app at: http://localhost:8501
    echo Press Ctrl+C to stop the server
    echo.
    streamlit run app/streamlit_app_enhanced.py
) else if "%choice%"=="3" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice. Exiting...
    pause
    exit /b 1
)
