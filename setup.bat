@echo off
echo ====================================================
echo   Instalador de Dependencias - IPP Dashboard
echo ====================================================
echo Inspecting Python environment...
python --version
echo.
echo Instalando bibliotecas necesarias (esto puede tardar un minuto)...
python -m pip install -r requirements.txt
echo.
echo ====================================================
echo   Instalacion completada con exito.
echo   Ahora puedes usar start_app.bat para lanzar la app.
echo ====================================================
pause
