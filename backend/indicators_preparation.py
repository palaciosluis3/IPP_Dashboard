import pandas as pd
import numpy as np
import os

# 1. Referencias flotantes para los archivos
# Asumimos que los archivos están en la misma carpeta que este script
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
for index, row in data.iterrows():
    time_series = row[years].values
    denominator = row.bestbound - row.worstbound
    
    # Evitar división por cero si bounds son iguales
    if denominator == 0:
        normalised_serie = np.zeros_like(time_series)
    else:
        normalised_serie = (time_series - row.worstbound) / denominator
    
    # --- NUEVA FUNCIONALIDAD: INTERPOLACIÓN ---
    # Convertimos a Series de Pandas y aseguramos que sea numérica
    s = pd.to_numeric(pd.Series(normalised_serie), errors='coerce')
    
    # Interpolación lineal para llenar los NaNs
    s = s.interpolate(method='linear', limit_direction='both')
    
    # Asegurar que los valores interpolados se mantengan entre [0, 1]
    s = s.clip(lower=0, upper=1)
    
    normalised_series.append(s.values)

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

# Valores iniciales y finales
df['I0'] = df[years[0]]
df['IF'] = df[years[-1]]

# Ajuste por si el indicador no cambió nada (I0 == IF)
# Se hace antes de calcular goals para asegurar que goals > IF final
df.loc[df.I0 == df.IF, 'IF'] = df.loc[df.I0 == df.IF, 'IF'] * 1.05

# 2. Loop para refinar la variable 'goals'
goals = []
for index, row in df.iterrows():
    # Obtenemos el target original
    raw_gov_target = data.loc[index, 'gov_target']
    
    # IMPORTANTE: El target del gobierno también debe normalizarse para ser comparable con IF
    denom = data.loc[index, 'bestbound'] - data.loc[index, 'worstbound']
    if denom != 0:
        norm_gov_target = (raw_gov_target - data.loc[index, 'worstbound']) / denom
    else:
        norm_gov_target = raw_gov_target

    # Lógica solicitada: si target > IF, tomar target; de lo contrario, IF * 1.05
    if norm_gov_target > row['IF']:
        goals.append(norm_gov_target)
    else:
        goals.append(row['IF'] * 1.05)

df['goals'] = goals

# Cálculo de tasas de éxito (Success Rates)
# Compara cada año con el anterior: sum(año_n > año_n-1) / total_transiciones
success_values = df[years].values
successRates = np.sum(success_values[:, 1:] > success_values[:, :-1], axis=1) / (len(years) - 1)

# Recomendación metodológica: evitar 0 y 1 puros
successRates[successRates == 0] = 0.05
successRates[successRates == 1] = 0.95
df['successRates'] = successRates

# Parámetros constantes (Gobernanza)
df['qm'] = QM_VALUE
df['rl'] = RL_VALUE

# Guardar resultado
print(f"Guardando en: {output_file}")
df.to_excel(output_file, index=False)
print("¡Proceso completado con éxito!")