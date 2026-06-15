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

# Factor de empujón mínimo usado en dos casos donde solo se busca romper una igualdad:
#  (a) metas ya alcanzadas (real_goal <= IF): PPI arroja error si la meta ya se cumplió
#      en la última observación, así que se fija una meta apenas por encima de IF.
#  (b) indicadores estáticos (I0 == IF): la calibración requiere I0 != IF.
# 1.01 = 1% por encima de IF (mínimo razonable para no simular un esfuerzo inexistente).
GOAL_INFLATION_FACTOR = 1.01

data = pd.read_excel(input_file)

# Identificar columnas de años (numéricas)
years = [column_name for column_name in data.columns if str(column_name).isnumeric()]

# REQUISITO IPP: los años deben ser CONSECUTIVOS (espaciamiento anual uniforme).
# Tanto la interpolación/extrapolación lineal como la calibración de PPI asumen pasos
# de tiempo equiespaciados; años con huecos (p. ej. 2015, 2018, 2020) producirían
# tasas de cambio e interdependencias incorrectas de forma silenciosa.
year_ints = [int(y) for y in years]
if any((year_ints[k + 1] - year_ints[k]) != 1 for k in range(len(year_ints) - 1)):
    raise ValueError(f"Los años deben ser consecutivos (sin huecos). Años detectados: {years}")

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

# 3. CÁLCULO DE MÉTRICAS (Sobre observaciones REALES, antes de interpolar)
# -------------------------------------------
# NOTA METODOLÓGICA: La interpolación NO debe contaminar estas métricas.
# I0, IF y successRates se calculan EXCLUSIVAMENTE con observaciones reales.
# La serie interpolada/extrapolada (df_interp) se usa únicamente para la red de
# interdependencias (interdependency_networks.py), que requiere series completas.

def first_valid(row_values):
    valid = row_values[~np.isnan(row_values)]
    return valid[0] if len(valid) else np.nan

def last_valid(row_values):
    valid = row_values[~np.isnan(row_values)]
    return valid[-1] if len(valid) else np.nan

# I0 = primera observación real; IF = última observación real (sin interpolar ni extrapolar)
df['I0'] = df[years].apply(lambda x: first_valid(x.values), axis=1)
df['IF'] = df[years].apply(lambda x: last_valid(x.values), axis=1)

# Cálculo de éxito sobre observaciones reales (ignora NaNs para ser fiel a los datos originales)
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

# --- VALIDACIÓN: número mínimo de observaciones reales ---
# IPP no admite indicadores con muy pocos datos (ni filas totalmente vacías).
# Exigimos al menos la mitad del horizonte temporal (p. ej. 5 de 10 años, 6 de 11/12).
min_required_obs = int(np.ceil(len(years) / 2))
valid_counts = df[years].notna().sum(axis=1)
insufficient_series = [
    (df.loc[index, 'seriesCode'], int(valid_counts.loc[index]))
    for index in df.index if valid_counts.loc[index] < min_required_obs
]

# --- INTERPOLACIÓN + EXTRAPOLACIÓN LINEAL (solo para la red de interdependencias) ---
# Interpolación lineal en huecos internos y extrapolación lineal en los extremos
# faltantes usando la pendiente del tramo más cercano (NO se repite la observación
# más cercana), acotada al intervalo ABIERTO (0, 1) de la escala normalizada.
# IMPORTANTE: IPP no acepta valores de 0 ni 1 puros, por eso recortamos a (EPS, 1-EPS).
EPS = 1e-6
def interpolate_extrapolate(row_values):
    y = np.array(row_values, dtype=float)
    n = len(y)
    x = np.arange(n)
    mask = ~np.isnan(y)
    xv, yv = x[mask], y[mask]
    if len(xv) == 0:
        return y  # serie vacía: la detiene la validación de observaciones mínimas
    if len(xv) == 1:
        return np.clip(np.full(n, yv[0]), EPS, 1.0 - EPS)
    out = np.interp(x, xv, yv)  # interior lineal (np.interp deja planos los extremos)
    # Extremo inicial: extrapolación lineal con la pendiente del primer tramo
    if xv[0] > 0:
        slope = (yv[1] - yv[0]) / (xv[1] - xv[0])
        out[:xv[0]] = yv[0] + slope * (x[:xv[0]] - xv[0])
    # Extremo final: extrapolación lineal con la pendiente del último tramo
    if xv[-1] < n - 1:
        slope = (yv[-1] - yv[-2]) / (xv[-1] - xv[-2])
        out[xv[-1] + 1:] = yv[-1] + slope * (x[xv[-1] + 1:] - xv[-1])
    # Límites duros del intervalo ABIERTO (0, 1): la extrapolación no puede tocar 0 ni 1
    return np.clip(out, EPS, 1.0 - EPS)

df_interp = pd.DataFrame(
    [interpolate_extrapolate(df.loc[index, years].values) for index in df.index],
    columns=years, index=df.index
)

# Indicadores sin cambio en el tiempo (I0 == IF).
# IPP SÍ los admite, pero la calibración requiere I0 != IF; por eso ajustamos IF.
# La idea es SOLO romper la igualdad, así que usamos el mismo empujón mínimo que las
# metas ya alcanzadas (GOAL_INFLATION_FACTOR), acotado para no sobrepasar el límite duro 1
# (si IF*factor >= 1, usamos el punto medio entre IF y 1, que conserva IF != I0 y < 1).
static_mask = df['I0'] == df['IF']
static_series = df.loc[static_mask, 'seriesCode'].tolist()
if static_series:
    if_static = df.loc[static_mask, 'IF']
    adjusted_IF = if_static * GOAL_INFLATION_FACTOR
    df.loc[static_mask, 'IF'] = np.where(adjusted_IF < 1.0, adjusted_IF, (if_static + 1.0) / 2.0)
    print(f"Nota: {len(static_series)} indicador(es) sin variación temporal; se ajustó IF (x{GOAL_INFLATION_FACTOR}) para permitir la calibración.")

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

    # Lógica: si la meta real supera el último nivel, usamos la meta real;
    # si ya está alcanzada (meta <= IF), fijamos una meta mínima por encima de IF
    # (acotada al intervalo abierto (0, 1) para no tocar el límite duro 1).
    if norm_gov_target > row['IF']:
        goals.append(norm_gov_target)
    else:
        goals.append(min(row['IF'] * GOAL_INFLATION_FACTOR, 1.0 - EPS))
    
    # Guardamos la meta real normalizada
    real_goals.append(norm_gov_target)

df['goals'] = goals
df['real_goals'] = real_goals

# --- REPORTE DE ERRORES DE VALIDACIÓN ---
# Nota: los indicadores estáticos (I0 == IF) NO son un error; IPP los admite y ya se
# ajustó su IF arriba. Solo detenemos por datos fuera de rango, metas fuera de rango
# o indicadores con observaciones insuficientes.
if out_of_bounds_series or out_of_bounds_targets or insufficient_series:
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

    if insufficient_series:
        print(f"\nLos siguientes indicadores ({len(insufficient_series)}) tienen menos de {min_required_obs} observaciones reales (mínimo = la mitad de {len(years)} años):")
        for code, cnt in insufficient_series:
            print(f" - {code}: {cnt} observación(es) válida(s)")

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