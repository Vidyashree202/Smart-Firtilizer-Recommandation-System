@echo off
echo Starting Smart Fertilizer Recommendation System with React Chatbot...
echo.

echo Starting Flask backend...
start "Flask Backend" cmd /k "cd /d %~dp0 && python app.py"

echo Waiting 3 seconds for Flask to start...
timeout /t 3 /nobreak > nul

echo Starting React chatbot...
start "React Chatbot" cmd /k "cd /d %~dp0\chatbot_integrated && npm run dev"

echo.
echo Both servers are starting...
echo Flask backend: http://127.0.0.1:5000
echo React chatbot: http://localhost:3000
echo Assistant page: http://127.0.0.1:5000/assistant
echo.
echo Press any key to exit...
pause > nul
