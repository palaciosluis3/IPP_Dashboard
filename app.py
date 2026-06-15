import streamlit as st
import os
import pandas as pd
import subprocess
import time
import re
import sys

# --- PARÁMETROS PERSISTENTES (Se actualizan al correr el modelo) ---
QM_PERSISTENT = 0.4248
RL_PERSISTENT = 0.5383
GROWTH_PERSISTENT = 25.0
YEARS_PERSISTENT = 15
INTER_YEAR_PERSISTENT = 4

# --- CONFIGURACIÓN DE PÁGINA Y ESTILO ---
st.set_page_config(
    page_title="IPP Dashboard | Análisis de Políticas Públicas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colores de la paleta solicitada
COLORS = {
    "deep_blue": "#1E3A8A",
    "sky_blue": "#3B82F6",
    "teal_green": "#0D9488",
    "amber": "#F59E0B",
    "charcoal": "#262730",
    "cool_gray": "#9CA3AF",
    "off_white": "#F9FAFB",
    "crimson_red": "#DC2626",
    "indigo": "#6366F1"
}

st.markdown(f"""
    <style>
    /* Estilos Generales - Respetando tema claro */
    .stApp {{
        background-color: {COLORS['off_white']};
        color: {COLORS['charcoal']};
    }}
    
    /* CABECERA: Reglas de máxima prioridad usando ID único */
    #ipp-header {{
        background-color: {COLORS['deep_blue']} !important;
        padding: 40px 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    #ipp-header h1, 
    #ipp-header div, 
    #ipp-header span, 
    #ipp-header p {{
        color: #FFFFFF !important;
        font-family: 'Source Sans Pro', sans-serif;
    }}
    
    #ipp-header h1 {{
        margin: 0 !important;
        padding: 0 !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
    }}
    
    #ipp-header .subtitle {{
        margin-top: 15px !important;
        font-size: 1.3rem !important;
        opacity: 0.95 !important;
        font-weight: 400 !important;
        color: #E0E7FF !important;
    }}

    /* CAJAS DE PASOS */
    .step-box {{
        background-color: white;
        padding: 25px;
        border-radius: 10px;
        border-left: 6px solid {COLORS['sky_blue']};
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        color: {COLORS['charcoal']};
    }}
    
    /* BOTONES */
    .stButton>button {{
        background-color: {COLORS['sky_blue']} !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 6px !important;
    }}
    .stButton>button:hover {{
        background-color: {COLORS['deep_blue']} !important;
        color: white !important;
    }}
    
    /* BOTÓN ATRÁS (Secundario) */
    div[data-testid="stFormSubmitButton"] > button, 
    .back-btn > div > button {{
        background-color: #F3F4F6 !important;
        color: {COLORS['charcoal']} !important;
        border: 1px solid {COLORS['cool_gray']} !important;
    }}

    /* LINKS */
    a {{
        color: {COLORS['sky_blue']} !important;
        text-decoration: none;
        font-weight: 600;
    }}
    
    /* TEXTOS GENERALES (Streamlit overrides) */
    .stApp label, .stMarkdown p {{
        color: {COLORS['charcoal']} !important;
    }}
    
    /* REGLA ESPECÍFICA PARA CARGADOR DE ARCHIVOS */
    [data-testid="stFileUploaderFileData"] {{
        color: {COLORS['charcoal']} !important;
    }}
    [data-testid="stFileUploaderFileData"] * {{
        color: {COLORS['charcoal']} !important;
    }}

    /* BARRA LATERAL */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['charcoal']};
    }}
    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}

    /* Estilo profesional para el área de arrastre de archivos (Dropzone) */
    [data-testid="stFileUploader"] section {{
        min-height: 220px !important;
        border: 2px dashed {COLORS['cool_gray']} !important;
        border-radius: 15px !important;
        padding: 40px !important;
        background-color: #f8f9fa !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
    }}
    
    [data-testid="stFileUploader"] section:hover {{
        border-color: {COLORS['sky_blue']} !important;
        background-color: #f0f7ff !important;
    }}

    [data-testid="stFileUploader"] section > div {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 10px !important;
        text-align: center !important;
        width: 100% !important;
    }}

    /* Ajuste del botón de examinar dentro del dropzone */
    [data-testid="stFileUploader"] button {{
        margin-top: 10px !important;
        width: auto !important;
    }}

    [data-testid="stFileUploader"] label {{
        display: block !important;
        text-align: center !important;
        width: 100% !important;
        margin-bottom: 10px !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- CABECERA RENDERIZADA ---
st.markdown(f"""
    <div id="ipp-header">
        <h1>📊 IPP Dashboard: Inferencia de Prioridades de Política</h1>
        <div class="subtitle">Sistema de Apoyo a Decisiones Estratégicas y Gasto Público</div>
    </div>
""", unsafe_allow_html=True)

# --- ESTADO DE LA APP ---
if 'step' not in st.session_state:
    st.session_state.step = 1

# Inicialización de parámetros para sincronización y persistencia
if 'qm' not in st.session_state: st.session_state.qm = QM_PERSISTENT
if 'rl' not in st.session_state: st.session_state.rl = RL_PERSISTENT
if 'annual_growth' not in st.session_state: st.session_state.annual_growth = GROWTH_PERSISTENT
if 'years_sim' not in st.session_state: st.session_state.years_sim = YEARS_PERSISTENT
if 'inter_year' not in st.session_state: st.session_state.inter_year = INTER_YEAR_PERSISTENT

if 'thresh' not in st.session_state: st.session_state.thresh = 0.90
if 'elastic' not in st.session_state: st.session_state.elastic = 0.03
if 'last_mile' not in st.session_state: st.session_state.last_mile = 0.90

if 'processing_done' not in st.session_state: st.session_state.processing_done = False

# Modo de ejecución: 'full' = pipeline completo, 'graphics' = solo regenerar gráficos
if 'mode' not in st.session_state: st.session_state.mode = 'full'

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1

# --- FUNCIONES DE UTILIDAD ---
def get_path(filename, folder=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    if folder:
        return os.path.join(base_path, folder, filename)
    # Por defecto, si es un archivo generado (Excel/PDF), buscamos en Outputs
    if filename.endswith('.xlsx') or filename.endswith('.pdf'):
        # Excepto los raw inputs
        if not filename.startswith('raw_'):
            return os.path.join(base_path, "Outputs", filename)
    return os.path.join(base_path, filename)

def update_script_config(file_path, replacements):
    """Actualiza las secciones de configuración de los scripts .py"""
    if not os.path.exists(file_path):
        return # Evitar errores si el archivo no está
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for target, replacement in replacements.items():
        # Busca patrones tipo VARIABLE = valor
        content = re.sub(rf"{target}\s*=\s*[\d\.]+", f"{target} = {replacement}", content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

# --- FLUJO DE LA APLICACIÓN ---

# Sidebar: Progreso y Referencias
with st.sidebar:
    st.title("Progreso del Proceso")
    if st.session_state.mode == 'graphics':
        # Flujo reducido: Inicio -> ODS -> Generación -> Resultados (pasos 1, 4, 5, 6)
        st.caption("Modo: Generación de gráficos únicamente")
        graphics_steps = [(1, "🏠 Inicio"), (4, "🎯 ODS a Graficar"), (5, "🚀 Generación"), (6, "📊 Resultados")]
        for step_num, label in graphics_steps:
            if st.session_state.step == step_num:
                st.markdown(f"**👉 {label}**")
            elif st.session_state.step > step_num:
                st.markdown(f"✅ {label}")
            else:
                st.markdown(f"⚪ {label}")
    else:
        steps_labels = ["📁 Carga de Datos", "🏛️ Gobernanza", "⚙️ Parámetros IPP", "🎯 ODS a Graficar", "🚀 Ejecución", "📊 Resultados"]
        for i, label in enumerate(steps_labels):
            if st.session_state.step == i + 1:
                st.markdown(f"**👉 {label}**")
            elif st.session_state.step > i + 1:
                st.markdown(f"✅ {label}")
            else:
                st.markdown(f"⚪ {label}")
    
    st.markdown("---")
    if st.button("🔄 Reiniciar Aplicación", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    # Espacio flexible para empujar las referencias hacia abajo
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # --- SECCIÓN DE REFERENCIAS (AGRUPADAS) ---
    st.markdown("---")
    st.markdown("#### 📚 Referencias")
    
    st.markdown(f"""
    <div style="font-size: 0.85rem; line-height: 1.4;">
        <p><b>Compilador App:</b><br>{'Luis Alberto Palacios'}</p>
        <p><b>Creadores IPP:</b><br>{'Dr. Omar Guerrero & Prof. Gonzalo Castañeda.'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<a href="https://policypriority.org/" target="_blank" style="font-size: 0.85rem;">🌐 Visitar policypriority.org</a>', unsafe_allow_html=True)

# --- PASO 1: CARGA DE ARCHIVOS ---
if st.session_state.step == 1:
    st.markdown('<div class="step-box"><h3>Paso 1: Carga de Archivos de Datos</h3><p>Por favor, sube los archivos Excel preparados para el análisis.</p></div>', unsafe_allow_html=True)
    
    # --- SECCIÓN DE TEMPLATES ---
    with st.expander("📥 ¿No tienes las plantillas? Descárgalas aquí", expanded=False):
        t_col1, t_col2 = st.columns(2)
        template_ind = get_path("raw_indicators.xlsx", folder="templates")
        template_exp = get_path("raw_expenditure.xlsx", folder="templates")
        
        with t_col1:
            if os.path.exists(template_ind):
                with open(template_ind, "rb") as f:
                    st.download_button("📝 Plantilla Indicadores", f, "raw_indicators.xlsx")
        with t_col2:
            if os.path.exists(template_exp):
                with open(template_exp, "rb") as f:
                    st.download_button("💰 Plantilla Presupuesto", f, "raw_expenditure.xlsx")
    st.markdown("---")

    # --- ATAJO: GENERACIÓN DE GRÁFICOS ÚNICAMENTE ---
    # Reutiliza la calibración/simulaciones ya guardadas en Outputs y salta directo
    # a la selección de ODS, sin volver a correr el motor completo.
    baseline_exists = os.path.exists(get_path("output_baseline.xlsx"))
    increase_exists = os.path.exists(get_path("output_increase.xlsx"))
    prev_run_exists = baseline_exists and increase_exists

    st.markdown('<div class="step-box" style="border-left-color:#0D9488;"><h4>⚡ Generación de gráficos únicamente</h4><p>¿Ya corriste el proceso completo antes? Cambia los ODS a graficar y regenera todas las gráficas y reportes <b>sin recalibrar ni volver a simular</b> (usa los resultados guardados en <code>Outputs</code>).</p></div>', unsafe_allow_html=True)
    if prev_run_exists:
        if st.button("📊 Ir a generar gráficos (usar resultados existentes)"):
            st.session_state.mode = 'graphics'
            st.session_state.step = 4
            st.rerun()
    else:
        st.info("ℹ️ Aún no hay resultados previos en `Outputs`. Primero ejecuta el proceso completo al menos una vez.")

    st.markdown("---")
    st.markdown('<h4>🚀 Proceso completo (calibración + simulación)</h4>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        ind_file = st.file_uploader("Sube raw_indicators.xlsx", type=["xlsx"])
    with col2:
        exp_file = st.file_uploader("Sube raw_expenditure.xlsx", type=["xlsx"])

    if ind_file and exp_file:
        if st.button("Guardar y Continuar ➡️"):
            st.session_state.mode = 'full'
            with open(get_path("raw_indicators.xlsx"), "wb") as f:
                f.write(ind_file.getbuffer())
            with open(get_path("raw_expenditure.xlsx"), "wb") as f:
                f.write(exp_file.getbuffer())
            st.success("Archivos guardados correctamente.")
            time.sleep(0.5)
            next_step()
            st.rerun()

# --- PASO 2: GOBERNANZA ---
elif st.session_state.step == 2:
    st.markdown('<div class="step-box"><h3>Paso 2: Indicadores de Gobernanza</h3><p>Define la calidad institucional del país según el Banco Mundial.</p></div>', unsafe_allow_html=True)
    
    st.info("💡 **Ayuda de Conversión:** El Banco Mundial entrega los datos en percentiles (0 a 100). Para el modelo IPP, usamos una escala de 0.0 a 1.0. **Ejemplo: Si el portal dice 48.11, introduce 0.4811.**")
    st.markdown(f'<a href="https://www.worldbank.org/en/publication/worldwide-governance-indicators/interactive-data-access" target="_blank">🌐 Worldwide Governance Indicators (Consulta aquí)</a>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        qm = st.number_input("QM - Control of Corruption (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=st.session_state.qm, format="%.4f")
    with col2:
        rl = st.number_input("RL - Rule of Law (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=st.session_state.rl, format="%.4f")
    
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("⬅️ Atrás"): prev_step(); st.rerun()
    with c2:
        if st.button("Aplicar y Continuar ➡️"):
            st.session_state.qm = qm
            st.session_state.rl = rl
            update_script_config(get_path("indicators_preparation.py", folder="backend"), {"QM_VALUE": qm, "RL_VALUE": rl})
            next_step()
            st.rerun()

# --- PASO 3: PARÁMETROS IPP ---
elif st.session_state.step == 3:
    st.markdown('<div class="step-box"><h3>Paso 3: Parámetros del Modelo IPP</h3><p>Configura las variables técnicas para la simulación y el reporte final.</p></div>', unsafe_allow_html=True)
    
    with st.expander("🛠️ Escenario Presupuestal y Tiempos", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            annual_growth = st.number_input("Crecimiento Anual del Presupuesto (en %)", value=st.session_state.annual_growth, min_value=0.0, max_value=100.0)
        with col2:
            years_sim = st.number_input("Años a proyectar (Total)", value=st.session_state.years_sim, min_value=1, max_value=50)
        
        total_growth_factor = 1.0 + (annual_growth / 100.0) * years_sim
        st.warning(f"📈 Resultado: Con un crecimiento anual del {annual_growth}%, el presupuesto final será **{total_growth_factor:.2f} veces** el inicial.")
        
        inter_year = st.number_input("Periodo Intermedio de Convergencia (Año)", value=st.session_state.inter_year, min_value=1, max_value=years_sim-1,
                                     help="Número de años en el cual desea analizar convergencias de indicadores tempranas a su metas establecidas.")
            
    with st.expander("🔬 Calibración y Diagnóstico", expanded=True):
        st.write("**Threshold de Calibración (Precisión esperada)**")
        st.session_state.thresh = st.slider(
            "Desliza para ajustar", 0.50, 0.95, value=st.session_state.thresh, step=0.01,
            help="Nivel de precisión deseado para la calibración del modelo."
        )
            
        st.write("**Umbral de Elasticidad (Respuesta al gasto)**")
        # El slider muestra % (1.0-5.0), guardamos decimal (0.01-0.05)
        elastic_pct = st.slider(
            "Desliza (%)", 1.0, 5.0, value=st.session_state.elastic * 100.0, step=0.1,
            help="Sensibilidad del indicador ante cambios en el gasto público."
        )
        st.session_state.elastic = elastic_pct / 100.0

        st.write("**Umbral de Última Milla (Proximidad a la meta)**")
        last_mile_pct = st.slider(
            "Desliza (%)", 80.0, 95.0, value=st.session_state.last_mile * 100.0, step=0.5,
            help="Cercanía mínima requerida para considerar que un indicador ha alcanzado su meta."
        )
        st.session_state.last_mile = last_mile_pct / 100.0

    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("⬅️ Atrás"): prev_step(); st.rerun()
    with c2:
        if st.button("Configurar Motor y Continuar ➡️"):
            st.session_state.annual_growth = annual_growth
            st.session_state.years_sim = years_sim
            st.session_state.inter_year = inter_year
            update_script_config(get_path("model_calibration.py", folder="backend"), {"threshold": st.session_state.thresh})
            update_script_config(get_path("prospective_simulation.py", folder="backend"), {"YEARS_TO_FORECAST": years_sim, "INTERMEDIATE_CONVERGENCE_YEAR": inter_year})
            update_script_config(get_path("prospective_simulation_increase.py", folder="backend"), {"YEARS_TO_FORECAST": years_sim, "INTERMEDIATE_CONVERGENCE_YEAR": inter_year, "BUDGET_GROWTH_FACTOR": total_growth_factor})
            update_script_config(get_path("prospective_simulation_byconsideration.py", folder="backend"), {"YEARS_TO_FORECAST": years_sim, "INTERMEDIATE_CONVERGENCE_YEAR": inter_year})
            update_script_config(get_path("graphics_only.py", folder="backend"), {"YEARS_TO_FORECAST": years_sim, "INTERMEDIATE_CONVERGENCE_YEAR": inter_year})
            update_script_config(get_path("final_report_generator.py", folder="backend"), {"ULTIMA_MILLA_THRESHOLD": st.session_state.last_mile, "ELASTICITY_THRESHOLD": st.session_state.elastic})
            next_step()
            st.rerun()

# --- PASO 4: ODS A GRAFICAR ---
elif st.session_state.step == 4:
    st.markdown('<div class="step-box"><h3>Paso 4: Selección de ODS a Graficar y Reportar</h3><p>Selecciona los Objetivos de Desarrollo Sostenible (ODS) cuyos indicadores deseas incluir en las gráficas y reportes finales. El modelo siempre correrá internamente con todo el set de indicadores para preservar las interdependencias.</p></div>', unsafe_allow_html=True)
    
    ODS_NAMES = {
        1: "ODS 1: Fin de la Pobreza",
        2: "ODS 2: Hambre Cero",
        3: "ODS 3: Salud y Bienestar",
        4: "ODS 4: Educación de Calidad",
        5: "ODS 5: Igualdad de Género",
        6: "ODS 6: Agua Limpia y Saneamiento",
        7: "ODS 7: Energía Asequible y No Contaminante",
        8: "ODS 8: Trabajo Decente y Crecimiento Económico",
        9: "ODS 9: Industria, Innovación e Infraestructura",
        10: "ODS 10: Reducción de las Desigualdades",
        11: "ODS 11: Ciudades y Comunidades Sostenibles",
        12: "ODS 12: Producción y Consumo Responsables",
        13: "ODS 13: Acción por el Clima",
        14: "ODS 14: Vida Submarina",
        15: "ODS 15: Vida de Ecosistemas Terrestres",
        16: "ODS 16: Paz, Justicia e Instituciones Sólidas",
        17: "ODS 17: Alianzas para lograr los Objetivos"
    }
    
    # Inicializar estado si no existe
    for ods_id in ODS_NAMES.keys():
        cb_key = f"ods_{ods_id}"
        if cb_key not in st.session_state:
            st.session_state[cb_key] = True
            
    # Función para marcar/desmarcar todos
    def toggle_all(select):
        for ods_id in ODS_NAMES.keys():
            st.session_state[f"ods_{ods_id}"] = select

    col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 7])
    with col_btn1:
        if st.button("Marcar Todos", use_container_width=True):
            toggle_all(True)
            st.rerun()
    with col_btn2:
        if st.button("Desmarcar Todos", use_container_width=True):
            toggle_all(False)
            st.rerun()
            
    cols = st.columns(3)
    selected_list = []
    for ods_id, ods_name in ODS_NAMES.items():
        col_idx = (ods_id - 1) % 3
        with cols[col_idx]:
            cb_key = f"ods_{ods_id}"
            val = st.checkbox(ods_name, key=cb_key)
            if val:
                selected_list.append(ods_id)
                
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("⬅️ Atrás"):
            # En modo gráficos, el paso anterior es el Inicio (paso 1)
            st.session_state.step = 1 if st.session_state.mode == 'graphics' else st.session_state.step - 1
            st.rerun()
    with c2:
        if st.button("Guardar Selección y Continuar ➡️"):
            if not selected_list:
                st.error("Debes seleccionar al menos un ODS para graficar.")
            else:
                import json
                selected_sdgs_file = get_path("selected_sdgs.json", folder="Outputs")
                os.makedirs(os.path.dirname(selected_sdgs_file), exist_ok=True)
                with open(selected_sdgs_file, 'w', encoding='utf-8') as f:
                    json.dump(selected_list, f)
                next_step()
                st.rerun()

# --- PASO 5: EJECUCIÓN ---
elif st.session_state.step == 5:
    is_graphics = st.session_state.mode == 'graphics'

    if is_graphics:
        st.markdown('<div class="step-box"><h3>Paso 5: Generación de Gráficos</h3><p>El sistema regenerará todas las gráficas y reportes para los ODS seleccionados, reutilizando la calibración y simulaciones ya guardadas en <code>Outputs</code> (sin recalibrar ni volver a simular).</p></div>', unsafe_allow_html=True)
        run_button_label = "📊 GENERAR GRÁFICOS"
    else:
        st.markdown('<div class="step-box"><h3>Paso 5: Ejecución del Motor IPP</h3><p>El sistema procesará los datos, calibrará el modelo y ejecutará las simulaciones.</p></div>', unsafe_allow_html=True)
        run_button_label = "🚀 INICIAR PROCESAMIENTO COMPLETO"

    # Listas de scripts según el modo
    GRAPHICS_SCRIPTS = [
        ("Generación de Gráficos", "graphics_only.py")
    ]
    FULL_SCRIPTS = [
        ("Prep. Indicadores", "indicators_preparation.py"),
        ("Redes Interdep.", "interdependency_networks.py"),
        ("Prep. Presupuesto", "expenditure_preparation.py"),
        ("CALIBRACIÓN", "model_calibration.py"),
        ("Simulación Base", "prospective_simulation.py"),
        ("Escenario Aumento", "prospective_simulation_increase.py"),
        ("Generador Reporte", "final_report_generator.py"),
        ("Gráficas por Consideración", "prospective_simulation_byconsideration.py")
    ]

    def execute_scripts(scripts):
        """Ejecuta secuencialmente la lista de scripts mostrando el log en vivo."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        # Creamos un contenedor dedicado para el log con altura fija
        log_container = st.empty()

        full_log = ""

        for i, (name, file) in enumerate(scripts):
            status_text.markdown(f"### ⚙️ {name}")

            # Ejecución capturando stdout y stderr en tiempo real de forma robusta
            process = subprocess.Popen(
                [sys.executable, "-u", get_path(file, folder="backend")],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                errors='replace' # Reemplaza caracteres que no puede decodificar en lugar de fallar
            )

            current_script_log = f"--- INICIANDO PROTOCOLO: {name.upper()} ---\n"

            # Leemos línea a línea para que Streamlit se actualice
            for line in iter(process.stdout.readline, ""):
                if line:
                    current_script_log += line
                    # Mostramos solo las últimas 20 líneas para evitar saturar la UI
                    lines = current_script_log.splitlines()
                    display_log = "\n".join(lines[-20:])
                    log_container.code(display_log)
                    full_log += line

            process.wait()
            if process.returncode != 0:
                st.error(f"❌ Error crítico en {name}. El proceso se detuvo.")
                st.warning("💡 Si el error menciona **'Permission denied'**, probablemente dejaste abierto un archivo de resultados (PDF o Excel) de la carpeta `Outputs`. Ciérralo e inténtalo de nuevo.")
                st.code(current_script_log) # Mostrar todo el log del error
                st.stop()

            progress_bar.progress((i + 1) / len(scripts))
            time.sleep(0.5)

        st.session_state.processing_done = True

    @st.dialog("⚠️ Cierra los archivos de resultados")
    def confirm_graphics_dialog():
        st.warning("Antes de regenerar, **cierra todos los archivos de resultados que tengas abiertos** de la carpeta `Outputs` (los PDF de gráficas, el reporte Excel, el resumen, etc.).")
        st.markdown("Si alguno está abierto, Windows impedirá sobrescribirlo y la generación **fallará**.")
        d1, d2 = st.columns(2)
        with d1:
            if st.button("❌ Cancelar", use_container_width=True):
                st.rerun()
        with d2:
            if st.button("✅ Confirmar y generar", use_container_width=True):
                st.session_state.run_graphics_confirmed = True
                st.rerun()

    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("⬅️ Atrás"): prev_step(); st.rerun()
    with c2:
        if st.button(run_button_label):
            if is_graphics:
                # Pedimos confirmación (modal) para recordar cerrar los archivos abiertos
                confirm_graphics_dialog()
            else:
                # Persistir parámetros en el código de app.py para la próxima sesión
                update_script_config(__file__, {
                    "QM_PERSISTENT": st.session_state.qm,
                    "RL_PERSISTENT": st.session_state.rl,
                    "GROWTH_PERSISTENT": st.session_state.annual_growth,
                    "YEARS_PERSISTENT": st.session_state.years_sim,
                    "INTER_YEAR_PERSISTENT": st.session_state.inter_year
                })
                execute_scripts(FULL_SCRIPTS)
                st.rerun()

    # Ejecución de gráficos una vez confirmado el modal
    if st.session_state.get('run_graphics_confirmed'):
        st.session_state.run_graphics_confirmed = False
        execute_scripts(GRAPHICS_SCRIPTS)
        st.rerun()

    if st.session_state.processing_done:
        st.success("✅ ¡Todo el procesamiento se completó con éxito!")
        if st.button("Ver Resultados ➡️"):
            st.session_state.processing_done = False # Reset para la próxima vez
            next_step()
            st.rerun()

# --- PASO 6: RESULTADOS ---
elif st.session_state.step == 6:
    st.markdown('<div class="step-box"><h3>Paso 6: Resultados y Reportes Ejecutivos</h3><p>Descarga tus archivos finales y consulta la carpeta de resultados para el análisis detallado.</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("⬇️ Descargas Directas")
        
        report_path = get_path("final_report_IPP.xlsx")
        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                st.download_button("📊 Descargar Reporte Excel Completo", f, "final_report_IPP.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        pdf_path = get_path("Resumen_Recomendaciones_IPP.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Descargar Resumen Ejecutivo (PDF)", f, "Resumen_Recomendaciones_IPP.pdf", "application/pdf")
        
        md_path = get_path("Resumen_Recomendaciones_IPP.md")
        if os.path.exists(md_path):
            with open(md_path, "rb") as f:
                st.download_button("📝 Descargar Resumen Ejecutivo (Markdown)", f, "Resumen_Recomendaciones_IPP.md", "text/markdown")

    
    with col2:
        st.subheader("📂 Carpeta de Resultados")
        outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Outputs")
        st.write("Todos los archivos generados y las gráficas se guardaron en:")
        st.code(outputs_dir)
        st.info("💡 **Nota:** Puedes abrir esta carpeta manualmente en tu explorador de Windows para revisar las gráficas y parámetros técnicos.")

    st.balloons()
