import os
import pandas as pd
import numpy as np
import requests
import warnings
import sys
import multiprocessing
import policy_priority_inference as ppi

# Desactivar advertencias
warnings.filterwarnings('ignore')

# 1. Configuración de Usuario (Escenario de Crecimiento Presupuestal)
# ---------------------------------------------------------
YEARS_TO_FORECAST = 16 
INTERMEDIATE_CONVERGENCE_YEAR = 4 
BUDGET_GROWTH_FACTOR = 5.0 
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
file_raw_indis = get_path('raw_indicators.xlsx')
file_params = get_path('parameters.xlsx')
file_net = get_path('data_network.xlsx')
file_exp = get_path('data_expenditure.xlsx')
file_rel = get_path('data_relational_table.xlsx')

# Carga de Datos
print(f"Iniciando simulación contrafactual (Crecimiento: {BUDGET_GROWTH_FACTOR}x)...")
df_indis = pd.read_excel(file_indis)
df_raw_indis = pd.read_excel(file_raw_indis)
df_params = pd.read_excel(file_params)
df_net = pd.read_excel(file_net)
df_exp = pd.read_excel(file_exp)
df_rela = pd.read_excel(file_rel)

N = len(df_indis)
I0 = df_indis.IF.values 
R = df_indis.instrumental.values
qm = df_indis.qm.values
rl = df_indis.rl.values
Imax = df_indis.maxVals.values
Imin = df_indis.minVals.values
goals = df_indis.goals.values

indis_index = {code: i for i, code in enumerate(df_indis.seriesCode)}
real_goals_map = dict(zip(df_raw_indis.seriesCode, df_raw_indis.gov_target))
real_goals = np.array([real_goals_map.get(code, 0) for code in df_indis.seriesCode])

alphas = df_params.alpha.values
alphas_prime = df_params.alpha_prime.values
betas = df_params.beta.values

A = np.zeros((N, N))
for _, row in df_net.iterrows():
    if row.origin in indis_index and row.destination in indis_index:
        A[indis_index[row.origin], indis_index[row.destination]] = row.weight

# --- Tiempos y Calibración ---
historical_years = [col for col in df_raw_indis.columns if str(col).isnumeric()]
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
    df_output.to_excel(get_path('output_increase.xlsx'), index=False)

    try:
        import matplotlib.pyplot as plt
        # VISUALIZACIONES 1 y 2: BARRAS (ESTÉTICA ORIGINAL)
        # ---------------------------------------------------------
        midpoint = 51 
        groups = [(0, midpoint, 'first'), (midpoint, N, 'second')]

        for start, end, label in groups:
            fig = plt.figure(figsize=(12, 4))
            subset = df_output.iloc[start:end]
            num_items = len(subset)
            for idx, (i, row) in enumerate(subset.iterrows()):
                x_pos = idx
                plt.bar(x_pos, row[0], color=row.color, width=.65, alpha=.5)
                # Flecha periodo intermedio
                inter_idx = INTERMEDIATE_CONVERGENCE_YEAR * calibration_index
                plt.arrow(x=x_pos, y=row[0], dx=0, dy=row[inter_idx]-row[0], color=row.color,
                          linewidth=2, alpha=1, head_width=.3, head_length=.02)
                # Flecha periodo final
                plt.arrow(x=x_pos, y=row[0], dx=0, dy=row[T_sim-1]-row[0], color=row.color,
                          linewidth=1, alpha=1, head_width=.3, head_length=.02, linestyle=':')
                # Meta Real
                plt.scatter(x_pos, row.real_goal, color='black', s=20, zorder=5)

            plt.xlim(-1, num_items)
            plt.gca().set_xticks(range(num_items))
            plt.gca().set_xticklabels(subset.seriesCode, rotation=90, fontsize=7)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.ylabel('levels', fontsize=14)
            plt.xlabel('indicators', fontsize=14)
            plt.tight_layout()
            plt.savefig(get_path(f'Bars_increase_{label}_half.pdf'))
            plt.close()

        # Visualización 3: Dona de Progreso (ESTÉTICA ORIGINAL)
        # ---------------------------------------------------------
        group0, group1, group2, group3, group4, group5 = [], [], [], [], [], []
        ids, colors = df_output.seriesCode, df_output.color
        for index, row in df_output.iterrows():
            if row[0] > 0:
                proportional_progress = 100*(row[T_sim-1]-row[0])/row[0]
                if proportional_progress <= 0: group0.append((ids[index], colors[index]))
                elif 0 < proportional_progress <= 5: group1.append((ids[index], colors[index]))
                elif 5 < proportional_progress <= 10: group2.append((ids[index], colors[index]))
                elif 10 < proportional_progress <= 20: group3.append((ids[index], colors[index]))
                elif 20 < proportional_progress <= 30: group4.append((ids[index], colors[index]))
                elif 30 < proportional_progress: group5.append((ids[index], colors[index]))

        fig = plt.figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(111)
        ax.axis('equal')
        width = 0.3
        pie, texts, pcts = ax.pie([len(group5), len(group4), len(group3), len(group2), len(group1), len(group0)], 
                                  radius=1-width, startangle=90, counterclock=False,
                                  colors=['whitesmoke','gainsboro','silver', 'darkgray','dimgrey', 'black'], 
                                  autopct='%.0f%%', pctdistance=0.79)
        plt.setp(pie, width=width, edgecolor='white')
        for i, p in enumerate(pcts): plt.setp(p, color='black' if i < 4 else 'white')
        ax.legend(pie, ['mayor a 30%', '20-30%', '10-20%','5-10%', '0-5%', 'Negativo'],
                  loc="center", bbox_to_anchor=(.25, .5, 0.5, .0), fontsize=7, frameon=False)

        cin = [c[1] for g in [group5, group4, group3, group2, group1, group0] for c in g]
        labels = [c[0] for g in [group5, group4, group3, group2, group1, group0] for c in g]
        pie2, _ = ax.pie(np.ones(len(labels)), radius=1, colors=cin, labels=labels, rotatelabels=True, 
                         counterclock=False, startangle=90, 
                         textprops=dict(va="center", ha='center', rotation_mode='anchor', fontsize=5, color='black'),
                         labeldistance=1.17)
        plt.setp(pie2, width=width, edgecolor='none')
        plt.tight_layout()
        plt.savefig(get_path('Donut_increase.pdf'))
        plt.close()

        # Visualización 4: Dona de Convergencia (CON DETECCIÓN DE MEJORAS)
        # ---------------------------------------------------------
        on_time, late, unfeasible = [], [], []
        all_vals = df_output.values[:, 3:3+T_sim]

        # 1. Determinar categorías del escenario actual (Aumento)
        for index, row in df_output.iterrows():
            meta = row.goal
            reaches = np.where(all_vals[index] >= meta)[0]
            if len(reaches) > 0 and reaches[0]/calibration_index <= INTERMEDIATE_CONVERGENCE_YEAR: on_time.append(index)
            elif len(reaches) > 0 and reaches[0]/calibration_index <= YEARS_TO_FORECAST - 1: late.append(index)
            else: unfeasible.append(index)

        # 2. Intentar cargar el escenario base para comparar "saltos"
        baseline_file = get_path('output_baseline.xlsx')
        jumpers = [] # Lista para guardar índices de indicadores que mejoraron
        if os.path.exists(baseline_file):
            try:
                df_base = pd.read_excel(baseline_file)
                base_vals = df_base.iloc[:, 3:3+T_sim].values
                base_goals = df_base['goal'].values
                
                for i in range(N):
                    r_base = np.where(base_vals[i] >= base_goals[i])[0]
                    cat_base = 0 if (len(r_base) > 0 and r_base[0]/calibration_index <= INTERMEDIATE_CONVERGENCE_YEAR) \
                               else 1 if (len(r_base) > 0 and r_base[0]/calibration_index <= YEARS_TO_FORECAST - 1) \
                               else 2
                    cat_inc = 0 if i in on_time else 1 if i in late else 2
                    if cat_inc < cat_base:
                        jumpers.append(i)
            except Exception as e:
                print(f"Nota: No se pudo comparar con baseline para detectar saltos: {e}")

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.axis('equal')
        pie, texts, pcts = ax.pie([len(on_time), len(late), len(unfeasible)], radius=1-width, startangle=90, counterclock=False,
                                  colors=['lightgrey', 'grey', 'black'], autopct='%.0f%%', pctdistance=0.79)
        plt.setp(pie, width=width, edgecolor='white')
        for i, p in enumerate(pcts): plt.setp(p, color='black' if i < 2 else 'white')

        # Cálculo robusto de años reales
        last_h_year_int = int(str(historical_years[-1]))
        year_conv = last_h_year_int + INTERMEDIATE_CONVERGENCE_YEAR
        year_final = last_h_year_int + YEARS_TO_FORECAST

        ax.legend(pie, [f'Llega a {year_conv}', 
                        f'{year_conv + 1}-{year_final}', f'> {year_final}'],
                  loc="center", bbox_to_anchor=(.25, .5, 0.5, .0), fontsize=7.8, frameon=False)

        indices_conv = on_time + late + unfeasible
        cin_conv = [df_output.loc[c].color for c in indices_conv]
        labels_conv = [df_output.loc[c].seriesCode for c in indices_conv]

        pie3, texts3 = ax.pie(np.ones(len(indices_conv)), radius=1, colors=cin_conv, labels=labels_conv, rotatelabels=True, 
                              counterclock=False, startangle=90, 
                              textprops=dict(va="center", ha='center', rotation_mode='anchor', fontsize=5, color='black'),
                              labeldistance=1.17)
        plt.setp(pie3, width=width, edgecolor='none')
        for i, idx_original in enumerate(indices_conv):
            if idx_original in jumpers:
                texts3[i].set_color('green')
                texts3[i].set_weight('bold')
                texts3[i].set_fontsize(5) # Mantenemos tamaño igual a los demás

        plt.tight_layout()
        plt.savefig(get_path('Donut_Convergencia_increase.pdf'))
        plt.close()
    except Exception as eppdf_inc:
        print(f"Error generando visualizaciones PDF (Increase): {eppdf_inc}")
        import traceback
        traceback.print_exc()

    print("\nSimulación Contrafactual y Visualizaciones (Originales) completadas exitosamente.")
