import pandas as pd
import numpy as np
import os

# 1. Referencias flotantes para los archivos
# Asumimos que los archivos están en la misma carpeta que este script
def get_path(filename):
    # Encontrar la raíz del proyecto (un nivel arriba de /backend)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_path = os.path.join(base_path, "backend")
    
    # 1. Si el archivo es un input crudo (raw_*), debe estar en la raíz
    if filename.startswith('raw_'):
        return os.path.join(base_path, filename)
    
    # 3. Todos los demás archivos (generados o intermedios) van a Outputs
    out_dir = os.path.join(base_path, "Outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)

input_file = get_path('raw_indicators.xlsx')
output_file = get_path('data_indicators.xlsx')

# 1. Configuración de Usuario (Indicadores de Gobernanza)
# Estos valores representan la calidad institucional promedio del país y afectan la eficiencia del gasto.
QM_VALUE = 0.4248 # Calidad del monitoreo
RL_VALUE = 0.5383 # Calidad del estado de derecho
data = pd.read_excel(input_file)

# Identificar columnas de años (numéricas)
years = [column_name for column_name in data.columns if str(column_name).isnumeric()]

# 3. Revisión de errores: Corregimos la normalización (faltaba el .append)
normalised_series = []
out_of_bounds_series = []
for index, row in data.iterrows():
    time_series = row[years].values
    denominator = row.bestbound - row.worstbound
    
    # 1. Alerta si bounds son iguales
    if denominator == 0:
        raise ValueError(f"Error: 'worstbound' y 'bestbound' son iguales para el indicador '{row.seriesCode}'. Esto no está permitido.")
    
    normalised_serie = (time_series - row.worstbound) / denominator
    
    # Convertimos a Series de Pandas para manejo flexible (pero sin interpolar aún)
    s = pd.to_numeric(pd.Series(normalised_serie), errors='coerce')
    
    # 2. Test para verificar que los datos normalizados estén estrictamente en (0, 1)
    # Lo hacemos sobre los datos disponibles (dropna) antes de interpolar
    if (s.dropna() <= 0).any() or (s.dropna() >= 1).any():
        out_of_bounds_series.append(row.seriesCode)
    
    normalised_series.append(s.values.copy())

# Crear el nuevo DataFrame con los datos normalizados
df = pd.DataFrame(normalised_series, columns=years)

# Copiar metadatos
df['seriesCode'] = data.seriesCode
df['sdg'] = data.sdg_target
df['minVals'] = np.zeros(len(data))
df['maxVals'] = np.ones(len(data))
df['instrumental'] = data.instrumental
df['seriesName'] = data.seriesName
df['color'] = data.color

# 3. CÁLCULO DE MÉTRICAS (Antes de interpolar)
# -------------------------------------------

# Valores iniciales y finales Crudos
df['I0'] = df[years[0]].copy()
df['IF'] = df[years[-1]].copy()

# Cálculo de éxito antes de interpolar (ignora NaNs para ser fiel a los datos originales)
def calculate_raw_success(row_values):
    valid = row_values[~np.isnan(row_values)]
    if len(valid) < 2: return 0.0
    return np.sum(valid[1:] > valid[:-1]) / (len(valid) - 1)

success_rates_list = df[years].apply(lambda x: calculate_raw_success(x.values), axis=1)
successRates = success_rates_list.values.copy()

# Recomendación metodológica: evitar 0 y 1 puros
successRates[successRates == 0] = 0.05
successRates[successRates == 1] = 0.95
df['successRates'] = successRates

# Primero creamos una versión interpolada para asegurar que cálculos de IF->goals sean robustos si falta el último año
df_interp = df[years].interpolate(method='linear', axis=1, limit_direction='both')
# Si I0 o IF son NaNs en el original, los completamos con la interpolación para no romper el modelo
df['I0'] = df['I0'].fillna(df_interp[years[0]])
df['IF'] = df['IF'].fillna(df_interp[years[-1]])

# Nuevo test: Identificar indicadores sin cambio (I0 == IF)
static_series = df[df.I0 == df.IF].seriesCode.tolist()

# Ajuste por si el indicador no cambió nada (I0 == IF)
df.loc[df.I0 == df.IF, 'IF'] = df.loc[df.I0 == df.IF, 'IF'] * 1.05

# 2. Loop para refinar la variable 'goals'
goals = []
real_goals = []
out_of_bounds_targets = []
for index, row in df.iterrows():
    # Obtenemos el target original
    raw_gov_target = data.loc[index, 'gov_target']
    
    # IMPORTANTE: El target del gobierno también debe normalizarse para ser comparable con IF
    denom = data.loc[index, 'bestbound'] - data.loc[index, 'worstbound']
    norm_gov_target = (raw_gov_target - data.loc[index, 'worstbound']) / denom

    # Test para verificar que el target normalizado esté estrictamente en (0, 1)
    if norm_gov_target <= 0 or norm_gov_target >= 1:
        out_of_bounds_targets.append((data.loc[index, 'seriesCode'], norm_gov_target))

    # Lógica solicitada: si target > IF, tomar target; de lo contrario, IF * 1.05
    if norm_gov_target > row['IF']:
        goals.append(norm_gov_target)
    else:
        goals.append(row['IF'] * 1.05)
    
    # Guardamos la meta real normalizada
    real_goals.append(norm_gov_target)

df['goals'] = goals
df['real_goals'] = real_goals

# --- REPORTE DE ERRORES DE VALIDACIÓN ---
if out_of_bounds_series or out_of_bounds_targets or static_series:
    print("\n" + "!" * 50)
    print("ERRORES DE VALIDACIÓN DETECTADOS")
    print("!" * 50)
    
    if out_of_bounds_series:
        print(f"\nLos siguientes indicadores ({len(out_of_bounds_series)}) tienen series temporales fuera de (0, 1):")
        for code in out_of_bounds_series:
            print(f" - {code}")
            
    if out_of_bounds_targets:
        print(f"\nLos siguientes indicadores ({len(out_of_bounds_targets)}) tienen metas (gov_target) fuera de (0, 1):")
        for code, val in out_of_bounds_targets:
            print(f" - {code}: valor normalizado = {val:.4f}")

    if static_series:
        print(f"\nLos siguientes indicadores ({len(static_series)}) presentan I0 == IF (sin cambio):")
        for code in static_series:
            print(f" - {code}")
            
    print("\nPor favor corrija los datos, bounds o los targets en el archivo Excel y vuelva a ejecutar.")
    print("!" * 50 + "\n")
    raise ValueError("El script se detuvo porque se encontraron inconsistencias en los datos.")

# --- FINALIZACIÓN ---
# Interpolamos las series temporales solo al final para entrega del archivo
df[years] = df_interp

# Parámetros constantes (Gobernanza)
df['qm'] = QM_VALUE
df['rl'] = RL_VALUE

# Guardar resultado
print(f"Guardando en: {output_file}")
df.to_excel(output_file, index=False)
print("¡Proceso completado con éxito!")