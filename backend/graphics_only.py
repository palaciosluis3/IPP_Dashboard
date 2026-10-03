import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from official_goals import (
    attach_official_goals, load_goal_sources, NO_GOAL_MESSAGE, convergence_note,
    CONVERGENCE_NOTE_FONT, CONVERGENCE_NOTE_SIZE, CONVERGENCE_NOTE_COLOR,
    CONVERGENCE_NOTE_GAP_INCHES,
)

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


def convergence_category(values, goal, calibration_index, intermediate_year, forecast_years):
    reaches = np.where(np.asarray(values, dtype=float) >= goal)[0]
    if len(reaches) and reaches[0] / calibration_index <= intermediate_year:
        return 0
    if len(reaches) and reaches[0] / calibration_index <= forecast_years - 1:
        return 1
    return 2


def finish_convergence_layout(fig, with_goal, without_goal):
    """Nota pequeña debajo del contenido, incluida al exportar con bbox='tight'."""
    fig.tight_layout()
    if without_goal:
        fig.canvas.draw()
        # El límite incluye las etiquetas rotadas: el pie queda por debajo de
        # todas ellas, con una separación de 0.16 pulgadas independientemente
        # del número de indicadores y de la longitud de sus nombres.
        content_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        footer_y = (content_bbox.y0 - CONVERGENCE_NOTE_GAP_INCHES) / fig.get_figheight()
        fig.text(0.5, footer_y,
                 convergence_note(with_goal, without_goal),
                 ha='center', va='top', fontfamily=CONVERGENCE_NOTE_FONT,
                 fontsize=CONVERGENCE_NOTE_SIZE, color=CONVERGENCE_NOTE_COLOR)


def plot_convergence(df, scenario, calibration_index, historical_years,
                     forecast_years, intermediate_year):
    # Retirar primero el gráfico anterior para evitar diagnósticos obsoletos.
    path = get_path(f'Donut_Convergencia_{scenario}.png')
    if os.path.exists(path):
        os.remove(path)
    evaluable = df.loc[df['has_official_goal']]
    groups = [[], [], []]
    time_cols = get_time_cols(df)
    for index, row in evaluable.iterrows():
        category = convergence_category(row[time_cols], row.real_goal,
                                        calibration_index, intermediate_year, forecast_years)
        groups[category].append(index)
    status = {
        'with_official_goal': len(evaluable),
        'without_official_goal': len(df) - len(evaluable),
        'on_time': len(groups[0]), 'late': len(groups[1]), 'unfeasible': len(groups[2]),
        'message': NO_GOAL_MESSAGE if evaluable.empty else
                   'Porcentajes calculados únicamente sobre indicadores con meta oficial.'
    }
    with open(get_path(f'convergence_{scenario}.json'), 'w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False)
    print(f"Convergencia ({scenario}): {len(evaluable)} con meta oficial; "
          f"{len(df)-len(evaluable)} excluidos sin meta oficial. {status['message']}")
    if evaluable.empty:
        return

    jumpers = []
    baseline_file = get_path('output_baseline.xlsx')
    if scenario == 'increase' and os.path.exists(baseline_file):
        df_base = attach_official_goals(pd.read_excel(baseline_file), *load_goal_sources(get_path))
        base_cols = get_time_cols(df_base)
        df_base = df_base.set_index('seriesCode')
        for category, indices in enumerate(groups):
            for index in indices:
                code = df.loc[index, 'seriesCode']
                if code in df_base.index and df_base.loc[code, 'has_official_goal']:
                    base_category = convergence_category(df_base.loc[code, base_cols],
                                                         df_base.loc[code, 'real_goal'],
                                                         calibration_index, intermediate_year, forecast_years)
                    if category < base_category:
                        jumpers.append(index)

    width = 0.3
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.axis('equal')
    pie, _, pcts = ax.pie([len(g) for g in groups], radius=1-width, startangle=90,
                         counterclock=False, colors=['lightgrey', 'grey', 'black'],
                         autopct='%.0f%%', pctdistance=0.79)
    plt.setp(pie, width=width, edgecolor='white')
    for i, p in enumerate(pcts):
        plt.setp(p, color='black' if i < 2 else 'white')
    last_year = int(str(historical_years[-1]))
    year_conv, year_final = last_year + intermediate_year, last_year + forecast_years
    ax.legend(pie, [f'Llega a {year_conv}', f'{year_conv + 1}-{year_final}', f'> {year_final}'],
              loc='center', bbox_to_anchor=(.25, .5, 0.5, .0), fontsize=7.8, frameon=False)
    indices = [i for g in groups for i in g]
    outer, labels = ax.pie(np.ones(len(indices)), radius=1,
                          colors=df.loc[indices, 'color'], labels=df.loc[indices, 'seriesCode'],
                          rotatelabels=True, counterclock=False, startangle=90,
                          textprops=dict(va='center', ha='center', rotation_mode='anchor', fontsize=5),
                          labeldistance=1.17)
    plt.setp(outer, width=width, edgecolor='none')
    for index, label in zip(indices, labels):
        if index in jumpers:
            label.set_color('green')
            label.set_weight('bold')
    finish_convergence_layout(fig, len(evaluable), len(df)-len(evaluable))
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_scenario(df_output, scenario, calibration_index, T_sim, historical_years,
                  forecast_years, intermediate_year):
    """Misma evaluación y presentación en el proceso completo y en gráficos."""
    df_output = attach_official_goals(df_output, *load_goal_sources(get_path))
    plot_convergence(df_output, scenario, calibration_index, historical_years,
                     forecast_years, intermediate_year)
    if df_output.empty:
        return
    time_cols = get_time_cols(df_output)
    inter_idx = min(intermediate_year * calibration_index, len(time_cols) - 1)
    num_plots = int(np.ceil(len(df_output) / 50))
    per_plot = int(np.ceil(len(df_output) / num_plots))
    for part in range(num_plots):
        subset = df_output.iloc[part*per_plot:(part+1)*per_plot]
        fig = plt.figure(figsize=(12, 4))
        for x_pos, (_, row) in enumerate(subset.iterrows()):
            start = row[time_cols[0]]
            plt.bar(x_pos, start, color=row.color, width=.65, alpha=.5)
            plt.arrow(x=x_pos, y=start, dx=0, dy=row[time_cols[inter_idx]]-start,
                      color=row.color, linewidth=2, alpha=1, head_width=.3, head_length=.02)
            plt.arrow(x=x_pos, y=start, dx=0, dy=row[time_cols[-1]]-start,
                      color=row.color, linewidth=1, alpha=1, head_width=.3, head_length=.02, linestyle=':')
            if row.has_official_goal:
                plt.scatter(x_pos, row.real_goal, color='black', s=20, zorder=5)
        plt.xlim(-1, len(subset))
        plt.xticks(range(len(subset)), subset.seriesCode, rotation=90, fontsize=7)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.ylabel('levels', fontsize=14)
        plt.xlabel('indicators', fontsize=14)
        plt.tight_layout()
        plt.savefig(get_path(f'Bars_{scenario}_part_{part+1}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Crecimiento: incluye también a los indicadores sin meta oficial.
    groups = [[] for _ in range(6)]
    for _, row in df_output.iterrows():
        start = row[time_cols[0]]
        if start > 0:
            progress = 100 * (row[time_cols[-1]] - start) / start
            category = next((i for i, upper in enumerate([0, 5, 10, 20, 30]) if progress <= upper), 5)
            groups[category].append((row.seriesCode, row.color))
    ordered = groups[::-1]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.axis('equal')
    width = .3
    pie, _, pcts = ax.pie([len(g) for g in ordered], radius=1-width, startangle=90,
                         counterclock=False, colors=['whitesmoke', 'gainsboro', 'silver', 'darkgray', 'dimgrey', 'black'],
                         autopct='%.0f%%', pctdistance=.79)
    plt.setp(pie, width=width, edgecolor='white')
    for i, p in enumerate(pcts):
        plt.setp(p, color='black' if i < 4 else 'white')
    ax.legend(pie, ['mayor a 30%', '20-30%', '10-20%', '5-10%', '0-5%', 'Negativo'],
              loc='center', bbox_to_anchor=(.25, .5, .5, .0), fontsize=7, frameon=False)
    items = [item for g in ordered for item in g]
    outer, _ = ax.pie(np.ones(len(items)), radius=1, colors=[c for _, c in items],
                      labels=[code for code, _ in items], rotatelabels=True, counterclock=False, startangle=90,
                      textprops=dict(va='center', ha='center', rotation_mode='anchor', fontsize=5), labeldistance=1.17)
    plt.setp(outer, width=width, edgecolor='none')
    plt.tight_layout()
    plt.savefig(get_path(f'Donut_{scenario}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_baseline(df_output, calibration_index, T_sim, historical_years,
                  forecast_years=None, intermediate_year=None):
    plot_scenario(df_output, 'baseline', calibration_index, T_sim, historical_years,
                  YEARS_TO_FORECAST if forecast_years is None else forecast_years,
                  INTERMEDIATE_CONVERGENCE_YEAR if intermediate_year is None else intermediate_year)


def plot_increase(df_output, calibration_index, T_sim, historical_years,
                  forecast_years=None, intermediate_year=None):
    plot_scenario(df_output, 'increase', calibration_index, T_sim, historical_years,
                  YEARS_TO_FORECAST if forecast_years is None else forecast_years,
                  INTERMEDIATE_CONVERGENCE_YEAR if intermediate_year is None else intermediate_year)


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
    plot_baseline(df_base_plot, calibration_index, T_sim, historical_years)

    # --- Increase ---
    df_inc_full = pd.read_excel(file_increase)
    T_sim_inc = len(get_time_cols(df_inc_full))
    df_inc_plot = filter_by_sdg(df_inc_full, selected_sdgs)
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
