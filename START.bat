@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Khong tao duoc moi truong. Cai Python 3.10-3.12 voi Tcl/Tk va py launcher.
        pause
        exit /b 1
    )
)
if not exist ".venv\installed.ok" (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Cai thu vien that bai. Kiem tra Internet roi chay lai.
        pause
        exit /b 1
    )
    type nul > ".venv\installed.ok"
)
".venv\Scripts\python.exe" face.py
if errorlevel 1 pause
