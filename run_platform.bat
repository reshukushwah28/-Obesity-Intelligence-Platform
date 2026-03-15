@echo off
echo Installing Dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 exit /b %errorlevel%

echo Training Models (This may take a few minutes)...
python src/train_platform.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo Launching App...
streamlit run app/streamlit_app.py
