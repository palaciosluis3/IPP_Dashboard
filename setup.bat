@echo off
set VENV_PATH=.venv
echo ====================================================
echo   Instalador de Dependencias - IPP Dashboard (VENV)
echo ====================================================

:: Verificar si el entorno virtual ya existe, si no, crearlo
if not exist %VENV_PATH% (
    echo Creando entorno virtual en %VENV_PATH%...
    python -m venv %VENV_PATH%
)

echo Activando entorno virtual...
call %VENV_PATH%\Scripts\activate

echo.
echo Actualizando pip e instalando bibliotecas necesarias...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ====================================================
echo   Instalacion completada con exito en entorno virtual.
echo   Ahora puedes usar start_app.bat para lanzar la app.
echo ====================================================
pause
