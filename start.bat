@echo off
rem Start the web app (dev mode). Requires dependencies installed and python in PATH.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python webapp.py
