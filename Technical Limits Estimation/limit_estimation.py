import pandas as pd
import numpy as np

# 1. Cargar la base de datos desde Excel
file_name = "raw_indicators.xlsx"
df = pd.read_excel(file_name)

# Aseguramos que los nombres de columnas sean strings para que .isnumeric() funcione
df.columns = [str(c) for c in df.columns]

# --- LÍNEA OFICIAL DEL AUTOR ADAPTADA ---
years = [column_name for column_name in df.columns if str(column_name).isnumeric()]
# ----------------------------------------

def clean_num(val):
    if pd.isna(val): return np.nan
    if isinstance(val, str):
        val = val.replace(',', '').strip()
    try: return float(val)
    except: return np.nan

new_rows = []

for index, row in df.iterrows():
    gov_target = clean_num(row.get('gov_target', np.nan))
    series_name = str(row.get('seriesName', '')).lower()
    
    # Identificar dirección (Invert: 1 = Negativo, 0 = Positivo)
    try:
        invert_val = int(clean_num(row.get('Invert', 0)))
    except:
        invert_val = 0
    is_positive = (invert_val == 0)
    
    # Extraer datos históricos usando la lista 'years' del autor
    hist_data = [clean_num(row[year]) for year in years]
    hist_series = pd.Series(hist_data).dropna()
    
    if hist_series.empty:
        new_row = row.copy()
        new_row['worstbound_suggested'] = np.nan
        new_row['bestbound_suggested'] = np.nan
        new_rows.append(new_row)
        continue
        
    hist_min = hist_series.min()
    hist_max = hist_series.max()
    hist_mean = hist_series.mean()
    spread = hist_max - hist_min
    
    # 2. Detección de tasas (Valores + Palabras clave en seriesName)
    is_rate = False
    has_keywords = any(kw in series_name for kw in ['tasa', 'porcentaje', '%'])
    
    # Detectar si la escala es 0-1 o 0-100
    if hist_min >= 0 and hist_max <= 100 and has_keywords:
        is_rate = True
        is_0_1 = (hist_max <= 1)
    
    # 3. Cálculo de márgenes (Regla 15% / 30% / 10%)
    rel_dispersion = spread / abs(hist_max) if hist_max != 0 else 0
    if rel_dispersion < 0.15:
        margin = max(spread * 0.30, abs(hist_mean) * 0.10)
    else:
        margin = spread * 0.10
        
    # 4. Límites sugeridos con expansión de meta al 10%
    target_expansion = 0.10
    
    if is_positive:
        new_worst = hist_min - margin
        new_best = hist_max + margin
        if not pd.isna(gov_target):
            new_best = max(new_best, gov_target + (margin if margin > 0 else abs(gov_target) * target_expansion))
            new_worst = min(new_worst, gov_target - margin)
    else:
        new_worst = hist_max + margin
        new_best = hist_min - margin
        if not pd.isna(gov_target):
            new_best = min(new_best, gov_target - (margin if margin > 0 else abs(gov_target) * target_expansion))
            new_worst = max(new_worst, gov_target + margin)
            
    # 5. Aplicar "Capping" si es tasa
    if is_rate:
        cap_max = 1.0 if is_0_1 else 100.0
        new_worst = max(0.0, min(cap_max, new_worst))
        new_best = max(0.0, min(cap_max, new_best))
    elif hist_min >= 0:
        if is_positive: new_worst = max(0.0, new_worst)
        else: new_best = max(0.0, new_best)
                
    new_row = row.copy()
    new_row['worstbound_suggested'] = new_worst
    new_row['bestbound_suggested'] = new_best
    new_rows.append(new_row)

# 6. Guardar en Excel
res_df = pd.DataFrame(new_rows)
res_df.to_excel('raw_indicators_suggested.xlsx', index=False)
output_filename = 'raw_indicators_suggested.xlsx'
print(f"¡Archivo generado con éxito: {output_filename}! Los límites calculados son una estimación y se sugiere su revisión cuidadosa.")