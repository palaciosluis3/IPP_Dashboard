import os
import pandas as pd
import numpy as np
import requests
import warnings
import sys
import policy_priority_inference as ppi

# Desactivar advertencias innecesarias
warnings.filterwarnings('ignore')

# 1. Referencias flotantes para los archivos
def get_path(filename):
    # Encontrar la raíz del proyecto (un nivel arriba de /backend)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_path = os.path.join(base_path, "backend")
    
    # 1. Si el archivo es un input crudo (raw_*), debe estar en la raíz
    if filename.startswith('raw_'):
        return os.path.join(base_path, filename)
    
    # 2. Todos los demás archivos (generados o intermedios) van a Outputs
    out_dir = os.path.join(base_path, "Outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)

# Archivos de entrada
file_indis = get_path('data_indicators.xlsx')
file_net = get_path('data_network.xlsx')
file_exp = get_path('data_expenditure.xlsx')
file_rel = get_path('data_relational_table.xlsx')
# Archivo de salida
file_params = get_path('parameters.xlsx')


# 2. Validación de archivos de entrada
files = [file_indis, file_net, file_exp, file_rel]
for f in files:
    if not os.path.exists(f):
        print(f"Error: No se encuentra el archivo {f}. Ejecuta los pasos previos.")
        sys.exit(1)

print("Cargando datos para la calibración...")

# Cargar bases de datos
df_indis = pd.read_excel(file_indis)
df_net = pd.read_excel(file_net)
df_exp = pd.read_excel(file_exp)
df_rela = pd.read_excel(file_rel)

# Prepara variables base
N = len(df_indis)
I0 = df_indis.I0.values
IF = df_indis.IF.values
success_rates = df_indis.successRates.values
R = df_indis.instrumental.values
qm = df_indis.qm.values
rl = df_indis.rl.values

# Mapeo de índices para la red
indis_index = {code: i for i, code in enumerate(df_indis.seriesCode)}

# Matriz de Adyacencia (Red de Interdependencia)
A = np.zeros((N, N))
for _, row in df_net.iterrows():
    if row.origin in indis_index and row.destination in indis_index:
        i = indis_index[row.origin]
        j = indis_index[row.destination]
        A[i, j] = row.weight

# Matriz de Presupuesto (Bs)
Bs = df_exp.values[:, 1:].astype(float)
T = Bs.shape[1]

# Tabla relacional (Diccionario de mapeo indicador -> programas de gasto)
B_dict = {}
for _, row in df_rela.iterrows():
    code = row.iloc[0]
    if code in indis_index:
        idx = indis_index[code]
        programmes = [p for p in row.values[1:] if pd.notna(p) and p != '']
        B_dict[idx] = programmes

# 3. Configuración de la Calibración
parallel_processes = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
threshold = 0.55
low_precision_counts = 50

print(f"Iniciando calibración del modelo...")
print(f"- Periodos (T): {T}")
print(f"- Procesos en paralelo: {parallel_processes}")
print(f"- Umbral de calidad: {threshold}")

# Ejecutar calibración
parameters = ppi.calibrate(
    I0, IF, success_rates, A=A, R=R, qm=qm, rl=rl, Bs=Bs, B_dict=B_dict,
    T=T, threshold=threshold, parallel_processes=parallel_processes, 
    verbose=True, low_precision_counts=low_precision_counts
)

# 4. Guardar resultados
if isinstance(parameters, np.ndarray) and parameters.ndim == 2:
    header = parameters[0]
    data = parameters[1:]
    df_params = pd.DataFrame(data, columns=header)
    for col in df_params.columns:
        df_params[col] = pd.to_numeric(df_params[col], errors='coerce')
    print(f"Calibración exitosa. Guardando resultados en: {file_params}")
    df_params.to_excel(file_params, index=False)
else:
    print("Error: La calibración no devolvió el formato esperado.")
    print(f"Tipo devuelto: {type(parameters)}")
    print(f"Contenido crudo: {parameters}")