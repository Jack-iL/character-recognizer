@echo off
rem One-click build (Windows)
rem Requirements: dependencies installed per README (CPU torch), python available in PATH
cd /d "%~dp0"
python -m PyInstaller --clean --noconfirm recognizer.spec
echo.
echo Build done. Output is in the "dist" folder.
pause
