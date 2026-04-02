# 📊 IPP Dashboard: Inferencia de Prioridades de Política

Esta aplicación es una herramienta avanzada basada en el modelo **Policy Priority Inference (PPI)**, diseñada para ayudar a los tomadores de decisiones a priorizar políticas públicas y optimizar el gasto público para alcanzar metas de desarrollo (ODS).

## ⬇️ Descargar la App

Puedes encontrar la última versión estable y las instrucciones de actualización en el repositorio oficial:
🔗 https://github.com/palaciosluis3/IPP_Dashboard/releases

## 🚀 Estructura del Proyecto

- `app.py`: Interfaz principal (ejecutada con Streamlit).
- `backend/`: Carpeta con los motores de cálculo y simulación.
- `Outputs/`: Carpeta donde se guardan automáticamente todos los Excel y reportes PDF generados.
- `raw_indicators.xlsx` y `raw_expenditure.xlsx`: Archivos de entrada con tus datos históricos y presupuestarios.

> ⚠️ **IMPORTANTE (Sincronización de Años):** El sistema es robusto al añadir más años (ej. 2025), pero la consistencia es clave. Si añades un nuevo año en tus indicadores, asegúrate de actualizarlo SIEMPRE en estos 4 puntos:
> 1. Columnas de años en `raw_indicators.xlsx`.
> 2. Columnas de años en hoja Presupuesto de `raw_expenditure.xlsx`.
> 3. Listado de años en hoja Población de `raw_expenditure.xlsx`.
> 4. Listado de años en hoja IPC de `raw_expenditure.xlsx`.

## 🛠️ Instalación y Configuración

Sigue estos pasos para poner en marcha la aplicación en tu computadora de forma aislada y segura:

0. Requiere Python 3.12 instalado en el sistema.

1. **Instalación Inicial (Aislamiento de Entorno)**:
   - Haz doble clic en el archivo `setup.bat`.
   - Este script creará automáticamente una carpeta llamada `.venv` que contiene una instancia de Python dedicada solo a esta app.
   - **Ventaja:** Esto garantiza que las librerías de la app no entren en conflicto con otras versiones de Python que ya tengas en tu sistema.
   - Solo necesitas ejecutarlo la primera vez o cuando se añadan nuevas librerías.

2. **Iniciar la Aplicación**:
   - Una vez instalado el entorno, haz doble clic en `start_app.bat` (o el acceso directo `PPI_Launcher` si está configurado).
   - El sistema detectará el entorno virtual y lanzará la interfaz de Streamlit en tu navegador predeterminado.

## 📖 Cómo usar la App

El flujo de trabajo está dividido en 5 pasos guiados:

1.  **Carga de Datos**: Sube tus archivos Excel `raw_indicators.xlsx` y `raw_expenditure.xlsx`.
2.  **Gobernanza**: Introduce los percentiles de Control de Corrupción (QM) y Estado de Derecho (RL) del Banco Mundial (en escala 0.0 a 1.0).
3.  **Parámetros IPP**: Configura el escenario de crecimiento presupuestal, los años a proyectar y los umbrales técnicos de calibración.
4.  **Ejecución**: Presiona el botón de procesamiento. El sistema usará **paralelización multicore** para calibrar el modelo y correr 1000 simulaciones Monte Carlo en tiempo récord.
5.  **Resultados**: Descarga el reporte final consolidado y el resumen ejecutivo en PDF con recomendaciones automáticas.

## 📚 Créditos y Referencias

- **Compilador de la App:** Luis Palacios
- **Creadores del modelo IPP:** Dr. Omar Guerrero & Prof. Gonzalo Castañeda.
- **Librería Python:** Esta app utiliza la librería oficial `policy-priority-inference`, mantenida por el equipo de IPP para asegurar rigor científico.
- **Más información:** Visita https://policypriority.org/ para conocer la metodología detallada detrás de este sistema. También, visita el repositorio oficial de IPP en GitHub: https://github.com/oguerrer/ppi.

---
*Desarrollado para potenciar la eficiencia en el diseño de políticas públicas basadas en evidencia.*
