@echo off
REM Expo React Native + Flask API Starter
REM Quick start script for Windows development

echo 🚀 Starting Expo React Native + Flask API Starter
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python first.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js first.
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm is not installed. Please install npm first.
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed
echo.

REM Start Flask server
echo 🐍 Starting Flask server...
cd server

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment and install dependencies
call venv\Scripts\activate.bat
echo Installing Python dependencies...
pip install -r requirements.txt

REM Copy environment file if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    copy env.example .env
)

REM Start Flask server in background
echo Starting Flask server on http://localhost:5000
start "Flask Server" cmd /k "venv\Scripts\activate.bat && python app.py"

REM Go back to root directory
cd ..

REM Start Expo app
echo.
echo 📱 Starting Expo React Native app...
cd mobile

REM Install dependencies
echo Installing Node.js dependencies...
npm install

REM Start Expo
echo Starting Expo development server...
echo Press 'w' for web, 'a' for Android, 'i' for iOS
echo.
npx expo start

pause





