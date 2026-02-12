import os
import pandas as pd
import numpy as np
import warnings
from PIL import Image, ImageDraw, ImageFont
import sys

# Desactivar advertencias
warnings.filterwarnings('ignore')

# 1. Configuración de Usuario
ULTIMA_MILLA_THRESHOLD = 0.9
ELASTICITY_THRESHOLD = 0.03
INTERMEDIATE_CONVERGENCE_YEAR = 4

def get_path(filename):
    # Encontrar la raíz del proyecto (un nivel arriba de /backend)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_path = os.path.join(base_path, "backend")
    
    # 1. Si es la librería PPI, se queda en backend
    if filename == 'policy_priority_inference.py':
        return os.path.join(backend_path, filename)
    
    # 2. Si el archivo es un input crudo (raw_*), debe estar en la raíz
    if filename.startswith('raw_'):
        return os.path.join(base_path, filename)
    
    # 3. Todos los demás archivos (generados o intermedios) van a Outputs
    out_dir = os.path.join(base_path, "Outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)

# Archivos
file_indis = get_path('data_indicators.xlsx')
file_raw_indis = get_path('raw_indicators.xlsx')
file_baseline = get_path('output_baseline.xlsx')
file_increase = get_path('output_increase.xlsx')
file_report = get_path('final_report_IPP.xlsx')
file_pdf = get_path('Resumen_Recomendaciones_IPP.pdf')

def generate_report():
    print("Iniciando generación de reporte final y tabla resumen...")
    
    if not all(os.path.exists(f) for f in [file_indis, file_raw_indis, file_baseline, file_increase]):
        print("Error: Faltan archivos necesarios. Asegúrate de correr todos los pasos previos.")
        return

    # 1. Cargar y Unificar Datos
    df_indis_prep = pd.read_excel(file_indis)
    df_raw = pd.read_excel(file_raw_indis)
    df_base = pd.read_excel(file_baseline).set_index('seriesCode')
    df_inc = pd.read_excel(file_increase).set_index('seriesCode')
    
    # Unificar todas las columnas de raw_indicators
    df_master = pd.merge(df_raw, df_indis_prep.drop(columns=['seriesName', 'color', 'instrumental', 'sdg'], errors='ignore'), on='seriesCode', how='left')

    # Calcular Índice de Calibración
    historical_years = [col for col in df_raw.columns if str(col).isnumeric()]
    if len(historical_years) > 0:
        calibration_index = int(np.round(50 / len(historical_years)))
        target_step = int(INTERMEDIATE_CONVERGENCE_YEAR * calibration_index)
        print(f"Calibración calculada: {calibration_index} pasos/año. Tiempo objetivo: paso {target_step} (Año {INTERMEDIATE_CONVERGENCE_YEAR})")
    else:
        calibration_index = 1
        target_step = INTERMEDIATE_CONVERGENCE_YEAR
        print("Advertencia: No se detectaron años históricos. Usando calibración por defecto (1).")

    final_rows = []
    
    # Categorías para el reporte visual
    cat_green = []
    cat_yellow = []
    cat_red = []

    for _, row in df_master.iterrows():
        code = row['seriesCode']
        name = row['seriesName']
        goal = row['goals']
        
        # Columnas de tiempo
        time_cols = [col for col in df_base.columns if str(col).startswith('t_') or str(col).isnumeric() or isinstance(col, int)]
        
        # Lógica de Cumplimiento
        try:
            base_series = df_base.loc[code, time_cols].values
            cumple_meta_baseline = 1 if any(base_series >= goal) else 0
            
            # NUEVO: Cumplimiento a tiempo (Gobierno)
            if target_step in df_base.columns:
                val_at_target = df_base.loc[code, target_step]
                cumple_a_tiempo = 1 if val_at_target >= goal else 0
            elif target_step < len(base_series):
                val_at_target = base_series[target_step]
                cumple_a_tiempo = 1 if val_at_target >= goal else 0
            else:
                cumple_a_tiempo = 0
            
            inc_series = df_inc.loc[code, time_cols].values
            cumple_meta_increase = 1 if any(inc_series >= goal) else 0
            
            # Elasticidad
            base_in, base_out = base_series[0], base_series[-1]
            growth_base = (base_out - base_in) / base_in if base_in > 0 else 0
            
            inc_in, inc_out = inc_series[0], inc_series[-1]
            growth_inc = (inc_out - inc_in) / inc_in if inc_in > 0 else 0
            
            elasticidad = 1 if (growth_inc - growth_base) > ELASTICITY_THRESHOLD else 0
        except KeyError:
            cumple_meta_baseline = 0
            cumple_a_tiempo = 0
            cumple_meta_increase = 0
            growth_base = 0
            growth_inc = 0
            elasticidad = 0

        ultima_milla = 1 if row['IF'] > ULTIMA_MILLA_THRESHOLD else 0
        
        # Recomendación
        cond_a_tiempo = (cumple_a_tiempo == 1)
        cond_no_elastico = (cumple_meta_baseline == 1 and elasticidad == 0)
        cond_no_instrumental = (cumple_meta_baseline == 1 and row['instrumental'] == 0)
        cond_ultima_milla = (ultima_milla == 1)

        if cond_a_tiempo or cond_no_elastico or cond_no_instrumental or cond_ultima_milla:
            recomendacion = "Continuar programas"
            cat_green.append(name)
        elif elasticidad == 1 and cumple_meta_increase == 1:
            recomendacion = "Escalar programas"
            cat_yellow.append(name)
        else:
            recomendacion = "Revisar los programas asociados"
            cat_red.append(name)
        
        res_row = row.to_dict()
        res_row.update({
            'Cumple Meta_baseline': cumple_meta_baseline,
            'Cumple_a_tiempo': cumple_a_tiempo,
            'Cumple Meta_increase': cumple_meta_increase,
            'Ultima_milla': ultima_milla,
            'Elastico': elasticidad,
            'Recomendacion_Final': recomendacion,
            'Crecimiento_baseline': growth_base,
            'Crecimiento_increase': growth_inc
        })
        final_rows.append(res_row)

    # Guardar Excel
    df_final = pd.DataFrame(final_rows)
    df_final.to_excel(file_report, index=False)
    print(f"Excel guardado en: {file_report}")

    # 2. Generación de Tabla Estética
    try:
        generate_visual_table(cat_green, cat_yellow, cat_red)
    except Exception as e:
        print(f"Error generando PDF: {e}")

def wrap_text(text, font, max_width, draw):
    words = str(text).split()
    lines = []
    if not words: return []
    current_line = words[0]
    for word in words[1:]:
        w, h = draw.textbbox((0, 0), current_line + " " + word, font=font)[2:]
        if w < max_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def generate_visual_table(green, yellow, red):
    print("Creando tabla visual estética (Layout Dinámico)...")
    
    # 1. Configuración de Estilo
    width = 1200
    margin = 50
    col_width = 220
    col_x = [280, 505, 730, 955]
    line_h = 15
    item_spacing = 2
    section_margin = 40
    
    # Fuentes
    try:
        font_bold = ImageFont.truetype("arialbd.ttf", 26)
        font_semi = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except:
        font_bold = ImageFont.load_default()
        font_semi = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Pre-calcular espacio necesario para cada sección
    temp_img = Image.new('RGB', (width, 100))
    temp_draw = ImageDraw.Draw(temp_img)
    
    section_data = [
        {"title": "Continuar con\nprogramas\nactuales", "color": (69, 119, 57), "items": green},
        {"title": "Programas\nescalables", "color": (255, 192, 0), "items": yellow},
        {"title": "Revisar diseño\ny/o\nimplementación\nen busca de\ncuellos de\nbotella", "color": (210, 34, 45), "items": red}
    ]

    total_needed_h = 150 # Margen inicial y final
    for sec in section_data:
        # Calcular cuántas líneas ocupa cada item
        sec_lines = []
        for item in sec["items"]:
            wrapped = wrap_text(item, font_small, col_width - 20, temp_draw)
            sec_lines.append(len(wrapped))
        
        # Distribuir en 4 columnas
        cols_heights = [0, 0, 0, 0]
        for lines in sec_lines:
            min_col = cols_heights.index(min(cols_heights))
            cols_heights[min_col] += (lines * line_h) + item_spacing
        
        # Altura de la sección = altura de la columna más alta + encabezado
        sec["h"] = max(150, max(cols_heights) + 100) # Mínimo 150px
        total_needed_h += sec["h"] + section_margin

    # 2. Crear Imagen con altura dinámica
    img = Image.new('RGB', (width, total_needed_h), color='white')
    draw = ImageDraw.Draw(img)
    
    current_y = 50
    for sec in section_data:
        y, h, title, col, items = current_y, sec["h"], sec["title"], sec["color"], sec["items"]
        
        # Caja principal
        draw.rounded_rectangle([30, y, width-30, y+h], radius=20, fill=(250, 250, 250), outline=(180, 180, 180), width=1)
        # Sidebar
        draw.rounded_rectangle([30, y, 250, y+h], radius=20, fill=col)
        draw.rectangle([230, y, 250, y+h], fill=col)
        
        # Texto Sidebar
        lines_title = title.split('\n')
        line_y = y + (h / 2) - (len(lines_title) * 20)
        for line in lines_title:
            draw.text((140, line_y), line, fill="white", font=font_bold, anchor="mm")
            line_y += 40
            
        # Listado dinámico
        draw.text((280, y + 20), "Indicadores seleccionados", fill="black", font=font_semi)
        
        cols_current_y = [y + 60] * 4
        for item in items:
            wrapped = wrap_text(item, font_small, col_width - 20, draw)
            # Elegir columna con menos altura actual
            c_idx = cols_current_y.index(min(cols_current_y))
            
            for i, line in enumerate(wrapped):
                prefix = "• " if i == 0 else "  "
                draw.text((col_x[c_idx], cols_current_y[c_idx]), prefix + line, fill=(60, 60, 60), font=font_small)
                cols_current_y[c_idx] += line_h
            cols_current_y[c_idx] += item_spacing
            
        current_y += h + section_margin

    img.save(file_pdf, "PDF", resolution=150.0)
    print(f"PDF generado dinámicamente: {file_pdf}")

if __name__ == "__main__":
    generate_report()
