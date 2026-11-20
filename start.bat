@echo off
title High-Volume OMR Scanner Server
echo ========================================================
echo  Starting High-Volume OMR Scanner (FastAPI + OpenCV)
echo  Port: 8000 | URL: http://127.0.0.1:8000
echo ========================================================
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause
