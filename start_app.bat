@echo off
set VENV_PATH=.venv

:: Verificar que el entorno virtual exista antes de lanzar
if not exist %VENV_PATH% (
    echo [ERROR] No existe el entorno virtual. Ejecute setup.bat primero.
    pause
    exit /b
)

echo Iniciando IPP Dashboard desde entorno virtual...
%VENV_PATH%\Scripts\python.exe -m streamlit run app.py
pause
