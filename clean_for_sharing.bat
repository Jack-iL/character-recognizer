@echo off
rem Clean a built CharacterRecognizer folder so it is safe to share.
rem Removes runtime files that contain YOUR local paths / logs.
rem Usage:  clean_for_sharing.bat  "path\to\dist\CharacterRecognizer"
rem         (without arguments it tries .\dist\CharacterRecognizer next to this script)

set TARGET=%~1
if "%TARGET%"=="" set TARGET=%~dp0dist\CharacterRecognizer

if not exist "%TARGET%" (
  echo Target folder not found: %TARGET%
  echo Usage: clean_for_sharing.bat "path\to\dist\CharacterRecognizer"
  pause
  exit /b 1
)

echo Cleaning: %TARGET%
echo   (removing runtime files that may contain your local paths)
if exist "%TARGET%\web_config.json" del /q "%TARGET%\web_config.json"
if exist "%TARGET%\config.txt"      del /q "%TARGET%\config.txt"
if exist "%TARGET%\debug_log.txt"   del /q "%TARGET%\debug_log.txt"
if exist "%TARGET%\hf_cache"        rmdir /s /q "%TARGET%\hf_cache"
if exist "%TARGET%\input_images"    rmdir /s /q "%TARGET%\input_images"
if exist "%TARGET%\output_results"  rmdir /s /q "%TARGET%\output_results"
if exist "%TARGET%\datasets"        rmdir /s /q "%TARGET%\datasets"

echo Done. This folder is now safe to zip and share.
echo The recipient will see a clean first run.
pause
