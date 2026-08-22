@echo off
setlocal
where python >nul 2>nul
if errorlevel 1 (
  echo Capybara needs Python 3.9+ installed and on PATH.
  echo Install it from https://www.python.org/downloads/
  exit /b 1
)
python "%~dp0gui.py" %*
endlocal
