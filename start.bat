@echo off
setlocal

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo ============================================
echo   Airbnb NYC Price Prediction - Startup
echo ============================================
echo.

:: Check Python is installed and in PATH
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Please install Python 3.8+ from https://www.python.org and try again.
    pause
    exit /b 1
)

echo [INFO] Python found:
python --version
echo.

:: Check dataset is present
if not exist "%PROJECT_DIR%AB_NYC_2019.csv" (
    echo [ERROR] Dataset not found: AB_NYC_2019.csv
    echo         Please ensure AB_NYC_2019.csv is in the project folder and try again.
    pause
    exit /b 1
)

echo [INFO] Dataset found: AB_NYC_2019.csv
echo.

:: Check if venv exists
if exist "%PROJECT_DIR%venv\Scripts\activate.bat" (
    echo [INFO] Found existing virtual environment.
    goto :activate
)

:: venv not found — create it and install dependencies
echo [INFO] No virtual environment found. Creating one...
python -m venv venv

if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    echo         Make sure Python is installed correctly and try again.
    pause
    exit /b 1
)

echo [INFO] Virtual environment created. Installing dependencies...
echo [INFO] Note: XGBoost and LightGBM may take a few minutes to install.
echo.
call "%PROJECT_DIR%venv\Scripts\activate.bat"
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies from requirements.txt.
    echo         Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo [INFO] Dependencies installed successfully.
goto :run

:activate
echo [INFO] Activating virtual environment...
call "%PROJECT_DIR%venv\Scripts\activate.bat"
echo [INFO] Virtual environment activated.
echo.

:run
echo [INFO] Starting Jupyter Notebook...
echo [INFO] The notebook will open in your browser automatically.
echo [INFO] Press Ctrl+C in this window to stop Jupyter.
echo.
jupyter notebook Airbnb_NYC_Price_Prediction.ipynb

echo.
echo [INFO] Jupyter stopped.
pause
