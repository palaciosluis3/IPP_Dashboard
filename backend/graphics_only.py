import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Desactivar advertencias
warnings.filterwarnings('ignore')

# 1. Configuración de Usuario (debe coincidir con prospective_simulation*.py)
# Estos valores los sincroniza app.py al configurar los parámetros del modelo.
# ---------------------------------------------------------
YEARS_TO_FORECAST = 15
INTERMEDIATE_CONVERGENCE_YEAR = 4
# ---------------------------------------------------------


def get_path(filename):
    # Encontrar la raíz del proyecto (un nivel arriba de /backend)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Si el archivo es un input crudo (raw_*), debe estar en la raíz
    if filename.startswith('raw_'):
        return os.path.join(base_path, filename)

    # 2. Todos los demás archivos (generados o intermedios) van a Outputs
    out_dir = os.path.join(base_path, "Outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def load_selected_sdgs():
    """Carga la lista de ODS seleccionados (por defecto, los 17)."""
    selected_sdgs = list(range(1, 18))
    selected_sdgs_file = get_path('selected_sdgs.json')
    if os.path.exists(selected_sdgs_file):
        with open(selected_sdgs_file, 'r') as f:
            selected_sdgs = [int(x) for x in json.load(f)]
    return selected_sdgs


def get_time_cols(df):
    """Devuelve las columnas de la serie temporal (0..T_sim-1), ordenadas."""
    time_cols = [c for c in df.columns
                 if isinstance(c, (int, np.integer)) or (isinstance(c, str) and c.isdigit())]
    return sorted(time_cols, key=lambda x: int(x))


def filter_by_sdg(df, selected_sdgs):
    """Filtra un dataframe de salida (con columna 'sdg') por los ODS seleccionados."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    links_path = os.path.join(base_dir, "SDG links.csv")
    if not os.path.exists(links_path):
        return df.reset_index(drop=True)

    df_links = pd.read_csv(links_path)
    target_to_sdg = dict(zip(df_links['SDG target'].astype(int), df_links['SDG'].astype(int)))
    df = df.copy()
    df['sdg_goal'] = df['sdg'].astype(int).map(target_to_sdg)
    df = df[df['sdg_goal'].isin(selected_sdgs)].drop(columns=['sdg_goal'])
    return df.reset_index(drop=True)


def get_calibration(historical_years):
    if len(historical_years) > 0:
        return int(np.round(50 / len(historical_years)))
    print("Aviso: No se detectaron años históricos. Usando calibración por defecto (1).")
    return 1


# ---------------------------------------------------------------------------
# GRÁFICAS DE BASELINE (réplica de prospective_simulation.py, usando real_goal)
# ---------------------------------------------------------------------------
def plot_baseline(df_output, calibration_index, T_sim, historical_years):
    print("Generando gráficas del escenario base...")

    # VISUALIZACIONES 1 y 2: BARRAS
    max_per_plot = 50
    num_plots = int(np.ceil(len(df_output) / max_per_plot))
    per_plot = int(np.ceil(len(df_output) / num_plots)) if num_plots > 0 else len(df_output)

    groups = []
    for i in range(num_plots):
        start = i * per_plot
        end = min((i + 1) * per_plot, len(df_output))
        groups.append((start, end, f'part_{i+1}'))

    for start, end, label in groups:
        fig = plt.figure(figsize=(12, 4))
        subset = df_output.iloc[start:end]
        num_items = len(subset)
        for idx, (i, row) in enumerate(subset.iterrows()):
            x_pos = idx
            plt.bar(x_pos, row[0], color=row.color, width=.65, alpha=.5)
            inter_idx = INTERMEDIATE_CONVERGENCE_YEAR * calibration_index
            plt.arrow(x=x_pos, y=row[0], dx=0, dy=row[inter_idx]-row[0], color=row.color,
                      linewidth=2, alpha=1, head_width=.3, head_length=.02)
            plt.arrow(x=x_pos, y=row[0], dx=0, dy=row[T_sim-1]-row[0], color=row.color,
                      linewidth=1, alpha=1, head_width=.3, head_length=.02, linestyle=':')
            plt.scatter(x_pos, row.real_goal, color='black', s=20, zorder=5)

        plt.xlim(-1, num_items)
        plt.gca().set_xticks(range(num_items))
        plt.gca().set_xticklabels(subset.seriesCode, rotation=90, fontsize=7)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.ylabel('levels', fontsize=14)
        plt.xlabel('indicators', fontsize=14)
        plt.tight_layout()
        plt.savefig(get_path(f'Bars_baseline_{label}.pdf'))
        plt.close()

    # Visualización 3: Dona de Progreso
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

    width = 0.3
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    pie, texts, pcts = ax.pie([len(group5), len(group4), len(group3), len(group2), len(group1), len(group0)],
                              radius=1-width, startangle=90, counterclock=False,
                              colors=['whitesmoke', 'gainsboro', 'silver', 'darkgray', 'dimgrey', 'black'],
                              autopct='%.0f%%', pctdistance=0.79)
    plt.setp(pie, width=width, edgecolor='white')
    for i, p in enumerate(pcts): plt.setp(p, color='black' if i < 4 else 'white')
    ax.legend(pie, ['mayor a 30%', '20-30%', '10-20%', '5-10%', '0-5%', 'Negativo'],
              loc="center", bbox_to_anchor=(.25, .5, 0.5, .0), fontsize=7, frameon=False)

    cin = [c[1] for g in [group5, group4, group3, group2, group1, group0] for c in g]
    labels = [c[0] for g in [group5, group4, group3, group2, group1, group0] for c in g]
    pie2, _ = ax.pie(np.ones(len(labels)), radius=1, colors=cin, labels=labels, rotatelabels=True,
                     counterclock=False, startangle=90,
                     textprops=dict(va="center", ha='center', rotation_mode='anchor', fontsize=5, color='black'),
                     labeldistance=1.17)
    plt.setp(pie2, width=width, edgecolor='none')
    plt.tight_layout()
    plt.savefig(get_path('Donut_baseline.pdf'))
    plt.close()

    # Visualización 4: Dona de Convergencia (usando la meta REAL)
    on_time, late, unfeasible = [], [], []
    all_vals = df_output.values[:, 3:3+T_sim]
    for index, row in df_output.iterrows():
        meta = row.real_goal if 'real_goal' in row else row.goal
        reaches = np.where(all_vals[index] >= meta)[0]
        if len(reaches) > 0 and reaches[0]/calibration_index <= INTERMEDIATE_CONVERGENCE_YEAR:
            on_time.append(index)
        elif len(reaches) > 0 and reaches[0]/calibration_index <= YEARS_TO_FORECAST - 1:
            late.append(index)
        else:
            unfeasible.append(index)

    total_cases = len(on_time) + len(late) + len(unfeasible)
    print(f"Diagnóstico Convergencia (base): A tiempo={len(on_time)}, Tarde={len(late)}, Inviable={len(unfeasible)}")

    if total_cases == 0:
        print("Advertencia: No hay datos suficientes para graficar la dona de convergencia base.")
        return

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    pie, texts, pcts = ax.pie([len(on_time), len(late), len(unfeasible)], radius=1-width, startangle=90, counterclock=False,
                              colors=['lightgrey', 'grey', 'black'], autopct='%.0f%%', pctdistance=0.79)
    plt.setp(pie, width=width, edgecolor='white')
    for i, p in enumerate(pcts): plt.setp(p, color='black' if i < 2 else 'white')

    last_h_year_int = int(str(historical_years[-1]))
    year_conv = last_h_year_int + INTERMEDIATE_CONVERGENCE_YEAR
    year_final = last_h_year_int + YEARS_TO_FORECAST
    ax.legend(pie, [f'Llega a {year_conv}', f'{year_conv + 1}-{year_final}', f'> {year_final}'],
              loc="center", bbox_to_anchor=(.25, .5, 0.5, .0), fontsize=7.8, frameon=False)

    indices_conv = on_time + late + unfeasible
    cin_conv = [df_output.loc[c].color for c in indices_conv]
    labels_conv = [df_output.loc[c].seriesCode for c in indices_conv]
    pie3, _ = ax.pie(np.ones(len(indices_conv)), radius=1, colors=cin_conv, labels=labels_conv, rotatelabels=True,
                     counterclock=False, startangle=90,
                     textprops=dict(va="center", ha='center', rotation_mode='anchor', fontsize=5, color='black'),
                     labeldistance=1.17)
    plt.setp(pie3, width=width, edgecolor='none')
    plt.tight_layout()
    plt.savefig(get_path('Donut_Convergencia_baseline.pdf'))
    plt.close()


# ---------------------------------------------------------------------------
# GRÁFICAS DE INCREASE (réplica de prospective_simulation_increase.py)
# ---------------------------------------------------------------------------
def plot_increase(df_output, calibration_index, T_sim, historical_years):
    print("Generando gráficas del escenario de aumento presupuestal...")

    # VISUALIZACIONES 1 y 2: BARRAS
    max_per_plot = 50
    num_plots = int(np.ceil(len(df_output) / max_per_plot))
    per_plot = int(np.ceil(len(df_output) / num_plots)) if num_plots > 0 else len(df_output)

    groups = []
    for i in range(num_plots):
        start = i * per_plot
        end = min((i + 1) * per_plot, len(df_output))
        groups.append((start, end, f'part_{i+1}'))

    for start, end, label in groups:
        fig = plt.figure(figsize=(12, 4))
        subset = df_output.iloc[start:end]
        num_items = len(subset)
        for idx, (i, row) in enumerate(subset.iterrows()):
            x_pos = idx
            plt.bar(x_pos, row[0], color=row.color, width=.65, alpha=.5)
            inter_idx = INTERMEDIATE_CONVERGENCE_YEAR * calibration_index
            plt.arrow(x=x_pos, y=row[0], dx=0, dy=row[inter_idx]-row[0], color=row.color,
                      linewidth=2, alpha=1, head_width=.3, head_length=.02)
            plt.arrow(x=x_pos, y=row[0], dx=0, dy=row[T_sim-1]-row[0], color=row.color,
                      linewidth=1, alpha=1, head_width=.3, head_length=.02, linestyle=':')
            plt.scatter(x_pos, row.real_goal, color='black', s=20, zorder=5)

        plt.xlim(-1, num_items)
        plt.gca().set_xticks(range(num_items))
        plt.gca().set_xticklabels(subset.seriesCode, rotation=90, fontsize=7)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.ylabel('levels', fontsize=14)
        plt.xlabel('indicators', fontsize=14)
        plt.tight_layout()
        plt.savefig(get_path(f'Bars_increase_{label}.pdf'))
        plt.close()

    # Visualización 3: Dona de Progreso
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

    width = 0.3
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    pie, texts, pcts = ax.pie([len(group5), len(group4), len(group3), len(group2), len(group1), len(group0)],
                              radius=1-width, startangle=90, counterclock=False,
                              colors=['whitesmoke', 'gainsboro', 'silver', 'darkgray', 'dimgrey', 'black'],
                              autopct='%.0f%%', pctdistance=0.79)
    plt.setp(pie, width=width, edgecolor='white')
    for i, p in enumerate(pcts): plt.setp(p, color='black' if i < 4 else 'white')
    ax.legend(pie, ['mayor a 30%', '20-30%', '10-20%', '5-10%', '0-5%', 'Negativo'],
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

    # Visualización 4: Dona de Convergencia (con detección de mejoras), usando meta REAL
    on_time, late, unfeasible = [], [], []
    all_vals = df_output.values[:, 3:3+T_sim]
    for index, row in df_output.iterrows():
        meta = row.real_goal if 'real_goal' in row else row.goal
        reaches = np.where(all_vals[index] >= meta)[0]
        if len(reaches) > 0 and reaches[0]/calibration_index <= INTERMEDIATE_CONVERGENCE_YEAR: on_time.append(index)
        elif len(reaches) > 0 and reaches[0]/calibration_index <= YEARS_TO_FORECAST - 1: late.append(index)
        else: unfeasible.append(index)

    # Comparar con el escenario base (alineado por seriesCode) para detectar "saltos"
    baseline_file = get_path('output_baseline.xlsx')
    jumpers = []
    if os.path.exists(baseline_file):
        try:
            df_base = pd.read_excel(baseline_file)
            df_base = df_base.set_index('seriesCode').reindex(df_output['seriesCode']).reset_index()
            base_vals = df_base.iloc[:, 3:3+T_sim].values
            base_goals = df_base['real_goal'].values
            for i in range(len(df_output)):
                r_base = np.where(base_vals[i] >= base_goals[i])[0]
                cat_base = 0 if (len(r_base) > 0 and r_base[0]/calibration_index <= INTERMEDIATE_CONVERGENCE_YEAR) \
                    else 1 if (len(r_base) > 0 and r_base[0]/calibration_index <= YEARS_TO_FORECAST - 1) \
                    else 2
                cat_inc = 0 if i in on_time else 1 if i in late else 2
                if cat_inc < cat_base:
                    jumpers.append(i)
        except Exception as e:
            print(f"Nota: No se pudo comparar con baseline para detectar saltos: {e}")

    width = 0.3
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    pie, texts, pcts = ax.pie([len(on_time), len(late), len(unfeasible)], radius=1-width, startangle=90, counterclock=False,
                              colors=['lightgrey', 'grey', 'black'], autopct='%.0f%%', pctdistance=0.79)
    plt.setp(pie, width=width, edgecolor='white')
    for i, p in enumerate(pcts): plt.setp(p, color='black' if i < 2 else 'white')

    last_h_year_int = int(str(historical_years[-1]))
    year_conv = last_h_year_int + INTERMEDIATE_CONVERGENCE_YEAR
    year_final = last_h_year_int + YEARS_TO_FORECAST
    ax.legend(pie, [f'Llega a {year_conv}', f'{year_conv + 1}-{year_final}', f'> {year_final}'],
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
            texts3[i].set_fontsize(5)
    plt.tight_layout()
    plt.savefig(get_path('Donut_Convergencia_increase.pdf'))
    plt.close()


def generate_graphics_only():
    print("\n" + "=" * 50)
    print("GENERACIÓN DE GRÁFICOS ÚNICAMENTE (reusando Outputs existentes)")
    print("=" * 50)

    file_baseline = get_path('output_baseline.xlsx')
    file_increase = get_path('output_increase.xlsx')
    file_indis = get_path('data_indicators.xlsx')

    faltantes = [f for f in [file_baseline, file_increase, file_indis] if not os.path.exists(f)]
    if faltantes:
        print("ERROR: Faltan archivos de una corrida previa. Debes ejecutar el proceso completo al menos una vez.")
        for f in faltantes:
            print(f" - No encontrado: {f}")
        raise SystemExit(1)

    selected_sdgs = load_selected_sdgs()
    print(f"ODS seleccionados: {selected_sdgs}")

    df_indis = pd.read_excel(file_indis)
    historical_years = [col for col in df_indis.columns if str(col).isnumeric()]
    calibration_index = get_calibration(historical_years)

    # --- Baseline ---
    df_base_full = pd.read_excel(file_baseline)
    T_sim = len(get_time_cols(df_base_full))
    df_base_plot = filter_by_sdg(df_base_full, selected_sdgs)
    if len(df_base_plot) == 0:
        print("Advertencia: ningún indicador del escenario base coincide con los ODS seleccionados.")
    else:
        plot_baseline(df_base_plot, calibration_index, T_sim, historical_years)

    # --- Increase ---
    df_inc_full = pd.read_excel(file_increase)
    T_sim_inc = len(get_time_cols(df_inc_full))
    df_inc_plot = filter_by_sdg(df_inc_full, selected_sdgs)
    if len(df_inc_plot) == 0:
        print("Advertencia: ningún indicador del escenario de aumento coincide con los ODS seleccionados.")
    else:
        plot_increase(df_inc_plot, calibration_index, T_sim_inc, historical_years)

    # --- Tabla final + PDF + Markdown (reutiliza la lógica existente) ---
    from final_report_generator import generate_report
    generate_report()

    # --- Gráficas por consideración (reutiliza la lógica existente) ---
    from prospective_simulation_byconsideration import generate_plots_by_consideration
    generate_plots_by_consideration()

    print("\nGeneración de gráficos (solo gráficos) completada exitosamente.")


if __name__ == "__main__":
    generate_graphics_only()
