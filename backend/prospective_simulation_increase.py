import os
import pandas as pd
import numpy as np
import requests
import warnings
import sys
import multiprocessing
import policy_priority_inference as ppi
from official_goals import official_goal_data

# Desactivar advertencias
warnings.filterwarnings('ignore')

# 1. Configuración de Usuario (Escenario de Crecimiento Presupuestal)
# ---------------------------------------------------------
YEARS_TO_FORECAST = 15 
INTERMEDIATE_CONVERGENCE_YEAR = 4 
BUDGET_GROWTH_FACTOR = 4.75 
# ---------------------------------------------------------

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

# Archivos
file_indis = get_path('data_indicators.xlsx')
file_params = get_path('parameters.xlsx')
file_net = get_path('data_network.xlsx')
file_exp = get_path('data_expenditure.xlsx')
file_rel = get_path('data_relational_table.xlsx')

# Carga de Datos
print(f"Iniciando simulación contrafactual (Crecimiento: {BUDGET_GROWTH_FACTOR}x)...")
df_indis = pd.read_excel(file_indis)
df_params = pd.read_excel(file_params)
df_net = pd.read_excel(file_net)
df_exp = pd.read_excel(file_exp)
df_rela = pd.read_excel(file_rel)

N = len(df_indis)
I0 = df_indis.IF.values.copy()
R = df_indis.instrumental.values.copy()
qm = df_indis.qm.values.copy()
rl = df_indis.rl.values.copy()
Imax = df_indis.maxVals.values.copy()
Imin = df_indis.minVals.values.copy()
goals = df_indis.goals.values.copy()

indis_index = {code: i for i, code in enumerate(df_indis.seriesCode)}
real_goals, has_official_goal = official_goal_data(df_indis)
real_goals = real_goals.values

alphas = df_params.alpha.values.copy()
alphas_prime = df_params.alpha_prime.values.copy()
betas = df_params.beta.values.copy()

A = np.zeros((N, N))
for _, row in df_net.iterrows():
    if row.origin in indis_index and row.destination in indis_index:
        A[indis_index[row.origin], indis_index[row.destination]] = row.weight

# --- Tiempos y Calibración ---
historical_years = [col for col in df_indis.columns if str(col).isnumeric()]
calibration_index = int(np.round(50 / len(historical_years)))
T_sim = YEARS_TO_FORECAST * calibration_index

print(f"Calibración detectada: {calibration_index}")
print(f"Simulando {YEARS_TO_FORECAST} años ({T_sim} cortes temporales)...")

# --- CONSTRUCCIÓN DEL PRESUPUESTO CONTRAFACTUAL ---
Bs_base = df_exp.values[:, 1:].astype(float)
Bs_tiles = np.tile(Bs_base[:, -1], (T_sim, 1)).T
linear_ramp = np.linspace(0, BUDGET_GROWTH_FACTOR - 1, T_sim)
growth_matrix = np.tile(linear_ramp, (Bs_tiles.shape[0], 1))
Bs = Bs_tiles * (1 + growth_matrix)

# Tabla Relacional
B_dict = {}
for _, row in df_rela.iterrows():
    code = row.iloc[0]
    if code in indis_index:
        B_dict[indis_index[code]] = [p for p in row.values[1:] if pd.notna(p) and p != '']

# Función auxiliar para paralelización
def run_single_simulation(kwargs):
    return ppi.run_ppi(**kwargs)

# Ejecución en paralelo
if __name__ == '__main__':
    sample_size = 1000 
    print(f"Corriendo {sample_size} simulaciones en paralelo ({multiprocessing.cpu_count()-1} núcleos)...")
    
    # Preparar argumentos con nombre para todas las simulaciones
    sim_kwargs = {
        'I0': I0, 'alphas': alphas, 'alphas_prime': alphas_prime, 'betas': betas,
        'A': A, 'R': R, 'qm': qm, 'rl': rl, 'Imax': Imax, 'Imin': Imin, 
        'Bs': Bs, 'B_dict': B_dict, 'T': T_sim, 'G': goals
    }
    all_kwargs = [sim_kwargs for _ in range(sample_size)]

    num_procs = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=num_procs) as pool:
        outputs = pool.map(run_single_simulation, all_kwargs)

    tsI, _, _, _, _, _ = zip(*outputs)
    tsI_hat = np.mean(tsI, axis=0)

    # Guardar Resultados
    output_columns = ['seriesCode', 'sdg', 'color'] + list(range(T_sim))
    new_rows = []
    for i, serie in enumerate(tsI_hat):
        new_row = [df_indis.iloc[i].seriesCode, df_indis.iloc[i].sdg, df_indis.iloc[i].color] + serie.tolist()
        new_rows.append(new_row)
    df_output = pd.DataFrame(new_rows, columns=output_columns)
    df_output['goal'] = goals
    df_output['real_goal'] = real_goals
    df_output['has_official_goal'] = has_official_goal.values

    # Guardar SIEMPRE el set completo de indicadores (sin filtrar por ODS).
    # Esto permite regenerar gráficos para cualquier selección de ODS sin recalibrar.
    df_output.to_excel(get_path('output_increase.xlsx'), index=False)

    # El filtro es solo para las gráficas; el Excel anterior conserva TODOS.
    from graphics_only import filter_by_sdg, load_selected_sdgs, plot_increase
    df_plot = filter_by_sdg(df_output, load_selected_sdgs())
    plot_increase(df_plot, calibration_index, T_sim, historical_years,
                    YEARS_TO_FORECAST, INTERMEDIATE_CONVERGENCE_YEAR)

    print("\nSimulación Contrafactual y Visualizaciones (Originales) completadas exitosamente.")
