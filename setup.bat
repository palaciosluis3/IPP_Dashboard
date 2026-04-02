@echo off
set VENV_PATH=.venv
echo ====================================================
echo    Instalador de Dependencias - IPP Dashboard (VENV)
echo ====================================================

:: Intentar crear el entorno virtual con Python 3.12 usando el lanzador 'py'
if not exist %VENV_PATH% (
    echo Detectada compatibilidad necesaria con Python 3.12...
    echo Creando entorno virtual en %VENV_PATH% usando Python 3.12...
    py -3.12 -m venv %VENV_PATH%
)

:: Verificación de seguridad: si falló la creación, no seguimos
if not exist %VENV_PATH%\Scripts\activate (
    echo [ERROR] No se pudo crear el entorno virtual. 
    echo Asegurate de que Python 3.12 este en 'py --list'.
    pause
    exit /b
)

echo Activando entorno virtual...
call %VENV_PATH%\Scripts\activate

echo.
echo Actualizando pip e instalando bibliotecas necesarias...
:: Una vez activado el venv, 'python' ya apunta al binario interno del entorno
python -m pip install --upgrade pip
python -m pip install setuptools
python -m pip install -r requirements.txt

echo.
echo ====================================================
echo    Instalacion completada con exito en entorno virtual.
echo    Ahora puedes usar start_app.bat para lanzar la app.
echo ====================================================
pause