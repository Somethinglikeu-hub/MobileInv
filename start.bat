@echo off
title BIST Picker Dashboard
cd /d "%~dp0"
python -m streamlit run bist_picker/dashboard/app.py --server.headless true
pause
