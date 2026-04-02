@echo off
set VENV_PATH=.venv
echo ====================================================
echo   Instalador de Dependencias - IPP Dashboard (VENV)
echo ====================================================

:: Intentar crear el entorno virtual con Python 3.12 para compatibilidad
if not exist %VENV_PATH% (
    echo Detectada compatibilidad necesaria con Python 3.12...
    echo Creando entorno virtual en %VENV_PATH% usando Python 3.12...
    python3.12 -m venv %VENV_PATH%
)

echo Activando entorno virtual...
call %VENV_PATH%\Scripts\activate

echo.
echo Actualizando pip e instalando bibliotecas necesarias...
python -m pip install --upgrade pip
python -m pip install setuptools
python -m pip install -r requirements.txt

echo.
echo ====================================================
echo   Instalacion completada con exito en entorno virtual.
echo   Ahora puedes usar start_app.bat para lanzar la app.
echo ====================================================
pause
