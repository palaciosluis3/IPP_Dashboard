# 📊 IPP Dashboard: Inferencia de Prioridades de Política

Esta aplicación es una herramienta avanzada basada en el modelo **Policy Priority Inference (PPI)**, diseñada para ayudar a los tomadores de decisiones a priorizar políticas públicas y optimizar el gasto público para alcanzar metas de desarrollo (ODS).

## ⬇️ Descargar la App

Descarga la aplicación completa desde el repositorio oficial:
🔗 https://github.com/palaciosluis3/IPP_Dashboard

En la rama `main`, pulsa **Code → Download ZIP** y extrae el archivo en tu computadora. No necesitas tener Git instalado. La distribución y el aviso de actualizaciones utilizan el repositorio directamente, sin Releases.

## 🔔 Actualizaciones

- Al abrir la app, se consulta una vez por sesión el `version.json` publicado en la rama `main`. Si su número es mayor que el de tu copia local, aparece una ventana con la versión disponible, las novedades y un enlace para descargar el ZIP completo.
- También puedes pulsar **Comprobar actualizaciones**, debajo de **Reiniciar Aplicación**, para repetir la consulta. El aviso no reaparece al cambiar de paso o descargar informes; puede volver a mostrarse al comprobar manualmente, reiniciar la aplicación o abrir una nueva sesión.
- La consulta requiere internet. Si falla la conexión o la información publicada no está disponible, puedes seguir trabajando y reintentar con el botón.
- Para actualizar, cierra la app, descarga y descomprime el ZIP. Abre la carpeta extraída y copia su contenido dentro de tu carpeta de instalación actual (donde está `start_app.bat`), aceptando reemplazar los archivos de la aplicación. Conserva tus archivos de datos (`raw_indicators.xlsx` y `raw_expenditure.xlsx`) y las carpetas `Outputs` y `.venv` sin borrarlos ni reemplazarlos, aunque aparezcan archivos con esos nombres en el ZIP. Después, abre `start_app.bat` como siempre. Así reutilizas tus datos, resultados y librerías instaladas.
- Solo necesitas ejecutar `setup.bat` de nuevo si las novedades indican cambios en las librerías. Este script reutiliza la carpeta `.venv` existente y actualiza sus dependencias.
- La app informa de la actualización; la descarga e instalación son manuales.

### Publicar una actualización (mantenimiento)

El archivo `version.json` de la raíz identifica la versión de cada copia descargada. La primera versión con este mecanismo es `1.0.0`. Para anunciar cambios:

1. Incrementa `version` con el formato numérico `X.Y.Z` (por ejemplo, `1.0.1` y después `1.0.2`). No uses sufijos como `-beta`. Las versiones se comparan numéricamente: `1.10.0` es posterior a `1.9.0`.
2. Actualiza `notes` con un resumen breve de las novedades. Es opcional y debe ser texto.
   Si cambian las dependencias de `requirements.txt`, indica en las novedades que se debe ejecutar `setup.bat` antes de abrir la app.
3. Publica `version.json` junto con todos los cambios de la app en la rama `main` de `palaciosluis3/IPP_Dashboard`, preferiblemente en el mismo commit. No incrementes únicamente el archivo remoto: el ZIP debe incluir el mismo número que se anuncia.

Ejemplo para la siguiente actualización:

```json
{
  "version": "1.0.1",
  "notes": "Correcciones y mejoras de los informes."
}
```

Solo las copias que ya incluyen este comprobador pueden mostrar avisos; quienes tengan una versión anterior deben descargarla manualmente una primera vez. La comprobación no envía datos de indicadores, presupuestos ni resultados.

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

> **Recomendación sobre las series históricas:** En lo posible, utiliza una base de indicadores simétrica, con series que cubran el mismo período y observaciones disponibles hasta el mismo año de corte. Esto facilita la interpretación de la calibración y del estado inicial de las simulaciones. Es una recomendación de calidad de datos, no un requisito adicional de la aplicación.

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

## Indicadores sin metas oficiales

- En `raw_indicators.xlsx`, deja vacía la celda `gov_target` cuando no exista una meta oficial. Si ningún indicador tiene meta, también puedes omitir esa columna. No uses textos como `NA`, `nan` o guiones para representar la ausencia.
- Cero es una meta ingresada, no una ausencia. Todas las metas ingresadas deben ser numéricas, finitas y normalizarse dentro de `(0, 1)` según `worstbound` y `bestbound`; un valor inválido detiene la preparación con el código del indicador y el valor problemático.
- Todos los indicadores se conservan en la calibración, las interdependencias, los vínculos presupuestarios y ambos escenarios, tengan o no meta. `has_official_goal` identifica las metas disponibles y `real_goals` conserva exclusivamente la meta oficial normalizada.
- Para mantener el criterio técnico vigente de PPI, un indicador sin meta usa `goals = min(IF × GOAL_INFLATION_FACTOR, 1 − EPS)` (factor predeterminado 1.01). Esta meta técnica se pasa como `G` a la simulación y no representa una meta gubernamental. No se modifica el modelo para usar `G=None`.
- Las barras y las donas de crecimiento incluyen todos los indicadores seleccionados. El punto de meta se dibuja solo si hay meta oficial. La convergencia excluye a quienes no la tienen: los porcentajes usan únicamente el subconjunto evaluable, con conteos de incluidos y excluidos. Si ninguno tiene meta, se indica “Convergencia no evaluable: no hay metas oficiales disponibles” y se retira cualquier dona de convergencia anterior.
- El Excel conserva niveles iniciales/finales, crecimientos, sensibilidad presupuestaria (`Elastico`) y última milla. Para indicadores sin meta, los tres campos de cumplimiento dicen `No evaluable`, `Estado_meta_oficial` indica la ausencia y `Recomendacion_Final` usa `Sin meta oficial`. `Meta_tecnica_simulacion` identifica explícitamente el objetivo técnico. PDF, Markdown y gráficas por consideración incluyen esta categoría: deben interpretarse las trayectorias, crecimiento, sensibilidad y última milla, sin asignar automáticamente Continuar, Escalar o Revisar.
- El modo de gráficos únicamente aplica las mismas reglas. En resultados antiguos sin bandera se consulta `real_goal` / `real_goals` o, si esas columnas faltan, las metas originales con sus bounds. `goal` / `goals` nunca prueban la existencia de una meta oficial.

Verificación reducida (datos sintéticos aislados de `Outputs`):
` .venv\Scripts\python.exe -m unittest discover -s tests -v `
Estas pruebas usan calibración reducida y tres simulaciones por escenario; sus parámetros y resultados son de prueba, no resultados calibrados para decisiones de política.

## 📚 Créditos y Referencias

- **Compilador de la App:** Luis Palacios
- **Creadores del modelo IPP:** Dr. Omar Guerrero & Prof. Gonzalo Castañeda.
- **Librería Python:** Esta app utiliza la librería oficial `policy-priority-inference`, mantenida por el equipo de IPP para asegurar rigor científico.
- **Más información:** Visita https://policypriority.org/ para conocer la metodología detallada detrás de este sistema. También, visita el repositorio oficial de IPP en GitHub: https://github.com/oguerrer/ppi.

---
*Desarrollado para potenciar la eficiencia en el diseño de políticas públicas basadas en evidencia.*
