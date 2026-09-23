@echo off
rem Release build: copies sources to a neutral path (C:\build_neutral) and builds there,
rem so the resulting exe does not embed any local username/paths.
rem Requirements: dependencies installed per README, python in PATH,
rem optional: run "python convert_fp16.py" first to halve the bundled model size.
cd /d "%~dp0"
set NEUTRAL=C:\build_neutral

echo [1/3] Preparing neutral build folder...
if exist "%NEUTRAL%" rmdir /s /q "%NEUTRAL%"
mkdir "%NEUTRAL%"
copy /y app_entry.py "%NEUTRAL%" >nul
copy /y webapp.py "%NEUTRAL%" >nul
copy /y recognizer.spec "%NEUTRAL%" >nul
xcopy /e /i /y static "%NEUTRAL%\static" >nul
if exist clip_model_fp16\model.safetensors (
  xcopy /e /i /y clip_model_fp16 "%NEUTRAL%\clip_model_fp16" >nul
) else if exist clip_model_fp16\pytorch_model.bin (
  xcopy /e /i /y clip_model_fp16 "%NEUTRAL%\clip_model_fp16" >nul
) else (
  xcopy /e /i /y clip_model "%NEUTRAL%\clip_model" >nul
)

echo [2/3] Building (this takes a few minutes)...
cd /d "%NEUTRAL%"
python -m PyInstaller --clean --noconfirm recognizer.spec

echo [3/3] Done
set OUT=%NEUTRAL%\dist\CharacterRecognizer
echo Output: %OUT%

echo Cleaning runtime files (so the output is safe to share)...
if exist "%OUT%\web_config.json" del /q "%OUT%\web_config.json"
if exist "%OUT%\config.txt"      del /q "%OUT%\config.txt"
if exist "%OUT%\debug_log.txt"   del /q "%OUT%\debug_log.txt"
if exist "%OUT%\hf_cache"        rmdir /s /q "%OUT%\hf_cache"
if exist "%OUT%\input_images"    rmdir /s /q "%OUT%\input_images"
if exist "%OUT%\output_results"  rmdir /s /q "%OUT%\output_results"
echo Share-ready: %OUT%
pause
