@echo off
echo Starting Smart Fertilizer Recommendation System (Single Terminal)
echo ================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    echo Please install Node.js and try again
    pause
    exit /b 1
)

REM Check if chatbot_integrated directory exists
if not exist "chatbot_integrated" (
    echo Error: chatbot_integrated directory not found
    echo Please make sure the React chatbot files are copied
    pause
    exit /b 1
)

REM Check if node_modules exists in chatbot_integrated
if not exist "chatbot_integrated\node_modules" (
    echo Installing React dependencies...
    cd chatbot_integrated
    npm install
    cd ..
    echo.
)

echo Starting both servers in single terminal...
echo.
python run_single_terminal.py

pause
