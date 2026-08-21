# -*- coding: utf-8 -*-
"""
graphics_from_report.py
=======================

Clon de `graphics_only.py` que se alimenta EXCLUSIVAMENTE de
`Outputs/final_report_IPP.xlsx`.

Motivo: el reporte fue armado pegando resultados de dos corridas de IPP
distintas (una de ellas fuera de la app), por lo que NO existen
`output_baseline.xlsx`, `output_increase.xlsx`, `data_indicators.xlsx` ni
`raw_indicators.xlsx`. Sin ellos `graphics_only.py` aborta en su chequeo de
pre-requisitos.

BASE DE LA RECONSTRUCCION
-------------------------
En `policy_priority_inference.py` el bucle principal hace `I = I0.copy()` y
luego `tsI[:, t] = I` ANTES de actualizar, de modo que `tsI[:, 0] == I0`. La
app inicializa la simulacion con `I0 = IF`. Por lo tanto:

    row[0]        == IF                              (exacto)
    row[T_sim-1]  == IF * (1 + Crecimiento)          (exacto)

ya que `final_report_generator.py` define
`growth = (series[-1] - series[0]) / series[0]`.

Esto permite reproducir sin aproximaciones la barra inicial, el nivel final y
las donas de progreso (`100 * Crecimiento`).

LO QUE NO SE PUEDE RECONSTRUIR
------------------------------
El nivel al anio 4 (`row[INTERMEDIATE_CONVERGENCE_YEAR * calibration_index]`),
es decir la flecha solida intermedia del grafico original: no esta en el Excel
ni se deriva de ninguna columna. En su lugar se dibuja UNA sola flecha
`IF -> nivel final` y la lectura de corto plazo se preserva en el punto de
meta (relleno = `Cumple_a_tiempo` 1, hueco = 0). No se fabrica ningun valor.

El script NUNCA escribe sobre `final_report_IPP.xlsx`: solo lo lee. Todos sus
productos van a `Outputs/graficas_reporte/`.
"""

import os
import re
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ---------------------------------------------------------
# 1. Configuracion de Usuario (debe coincidir con prospective_simulation*.py)
# Estos valores los sincroniza app.py al configurar los parametros del modelo.
# NO renombrar (regla 2 de AGENTS.md).
# ---------------------------------------------------------
YEARS_TO_FORECAST = 15
INTERMEDIATE_CONVERGENCE_YEAR = 4
# ---------------------------------------------------------

OUTPUT_SUBDIR = "graficas_reporte"
INPUT_FILENAME = "final_report_IPP.xlsx"

# Contrato de columnas
REQUIRED_COLS = ['seriesCode', 'color', 'IF', 'Crecimiento_baseline']
COL_GOAL = 'real_goals'
COL_ONTIME = 'Cumple_a_tiempo'
COL_MEET_BASE = 'Cumple Meta_baseline'      # ojo: espacio, no guion bajo
COL_MEET_INC = 'Cumple Meta_increase'       # ojo: espacio, no guion bajo
COL_GROWTH_BASE = 'Crecimiento_baseline'
COL_GROWTH_INC = 'Crecimiento_increase'
COL_RECO = 'Recomendacion_Final'

RECOMMENDATIONS = [
    "Continuar programas",
    "Escalar programas",
    "Revisar los programas asociados",
]
FILE_SUFFIXES = {
    "Continuar programas": "continuar",
    "Escalar programas": "escalar",
    "Revisar los programas asociados": "revisar",
}

DONUT_WIDTH = 0.3
SIN_META_COLOR = '#D9D2E9'

# Categorias de convergencia: a tiempo / tarde / inviable / sin meta
CONV_COLORS = ['lightgrey', 'grey', 'black', SIN_META_COLOR]
CONV_HATCHES = [None, None, None, '//']
CONV_LEGEND_FONTSIZE = 6.5


# ---------------------------------------------------------------------------
# HELPERS DE RUTAS
# ---------------------------------------------------------------------------
def _base_dir():
    """Raiz del proyecto (un nivel arriba de /backend)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_path(filename):
    """Ruta de SALIDA: Outputs/graficas_reporte/<filename>."""
    out_dir = os.path.join(_base_dir(), "Outputs", OUTPUT_SUBDIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def get_input_path(filename):
    """Ruta de ENTRADA: Outputs/<filename> (o la raiz para los raw_*)."""
    base_path = _base_dir()
    if filename.startswith('raw_'):
        return os.path.join(base_path, filename)
    return os.path.join(base_path, "Outputs", filename)


def load_selected_sdgs():
    """Carga la lista de ODS seleccionados (por defecto, los 17)."""
    selected_sdgs = list(range(1, 18))
    selected_sdgs_file = get_input_path('selected_sdgs.json')
    if os.path.exists(selected_sdgs_file):
        with open(selected_sdgs_file, 'r') as f:
            selected_sdgs = [int(x) for x in json.load(f)]
    return selected_sdgs


def filter_by_sdg(df, selected_sdgs):
    """Filtra el reporte por ODS usando 'sdg_target' y 'SDG links.csv'.

    El reporte trae `sdg_target` (1..169), no la columna `sdg` de los outputs
    de simulacion. Se mapea a ODS (1..17) igual que el resto del pipeline.
    """
    if 'sdg_target' not in df.columns:
        print("  [i] Sin columna 'sdg_target': se omite el filtro por ODS.")
        return df.reset_index(drop=True)

    links_path = os.path.join(_base_dir(), "SDG links.csv")
    if not os.path.exists(links_path):
        print("  [i] No se encontro 'SDG links.csv': se omite el filtro por ODS.")
        return df.reset_index(drop=True)

    df_links = pd.read_csv(links_path)
    target_to_sdg = dict(zip(df_links['SDG target'].astype(int),
                             df_links['SDG'].astype(int)))
    df = df.copy()
    df['sdg_goal'] = df['sdg_target'].astype(int).map(target_to_sdg)

    sin_mapeo = df.loc[df['sdg_goal'].isna(), 'sdg_target'].tolist()
    if sin_mapeo:
        print(f"  [!] sdg_target sin correspondencia en SDG links.csv: {sin_mapeo}")

    df = df[df['sdg_goal'].isin(selected_sdgs)].drop(columns=['sdg_goal'])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# HELPERS DE DATOS
# ---------------------------------------------------------------------------
def get_year_columns(df):
    """Devuelve [(anio_int, nombre_columna)] de las series historicas crudas.

    Acepta '2019_x' (sufijo de la fusion de pandas en final_report_generator),
    '2019' a secas y, como ultimo recurso, '2019_y' (serie normalizada).
    """
    for pattern in (r'^(\d{4})_x$', r'^(\d{4})$', r'^(\d{4})_y$'):
        found = []
        for c in df.columns:
            m = re.match(pattern, str(c))
            if m:
                found.append((int(m.group(1)), c))
        if found:
            return sorted(found)
    return []


def last_observed_year(df, mask=None):
    """Ultimo anio con observacion real; None si no hay columnas de anio."""
    year_cols = get_year_columns(df)
    if not year_cols:
        return None
    sub = df if mask is None else df[mask]
    if len(sub) == 0:
        return None
    last = None
    for year, col in year_cols:
        if sub[col].notna().any():
            last = year if last is None else max(last, year)
    return last


def convergence_labels(df, has_goal):
    """Etiquetas de las 4 categorias de convergencia, fechadas cuando se puede.

    El anio de corte sale del ultimo anio observado ENTRE LAS FILAS CON META,
    que son las unicas que entran en la clasificacion temporal.
    """
    last_year = last_observed_year(df, pd.Series(has_goal, index=df.index))
    if last_year is not None:
        y_conv = last_year + INTERMEDIATE_CONVERGENCE_YEAR
        y_final = last_year + YEARS_TO_FORECAST
        return [f'Llega a {y_conv}', f'{y_conv + 1}-{y_final}',
                f'> {y_final}', 'Sin meta']
    return [f'Llega al anio {INTERMEDIATE_CONVERGENCE_YEAR}',
            f'Anios {INTERMEDIATE_CONVERGENCE_YEAR + 1}-{YEARS_TO_FORECAST}',
            f'> anio {YEARS_TO_FORECAST}', 'Sin meta']


def reconstruct_levels(df, growth_col):
    """(inicio, fin) de la trayectoria simulada, reconstruidos desde el reporte.

    inicio = IF                     (== row[0] del output_*.xlsx)
    fin    = IF * (1 + Crecimiento) (== row[T_sim-1])
    """
    start = df['IF'].astype(float).values
    growth = df[growth_col].astype(float).values
    return start, start * (1.0 + growth)


def goal_info(df, stages):
    """Vectores auxiliares de meta: (tiene_meta, valor_meta, a_tiempo)."""
    n = len(df)
    if not stages['has_goal']:
        return np.zeros(n, dtype=bool), np.full(n, np.nan), np.zeros(n, dtype=bool)

    goal = df[COL_GOAL].astype(float).values
    has_goal = ~np.isnan(goal)

    if stages['has_ontime']:
        on_time = (df[COL_ONTIME].fillna(0).astype(float).values == 1) & has_goal
    else:
        on_time = np.zeros(n, dtype=bool)

    return has_goal, goal, on_time


# ---------------------------------------------------------------------------
# PREFLIGHT DE DISPONIBILIDAD
# ---------------------------------------------------------------------------
def check_availability(df):
    """Clasifica columnas y decide que etapas pueden correr.

    Falla solo si faltan las columnas obligatorias; cualquier otra ausencia
    degrada la etapa correspondiente en vez de detener el script.
    """
    print("\n--- Revision de disponibilidad de datos ---")

    faltantes = [c for c in REQUIRED_COLS if c not in df.columns]
    if faltantes:
        print("ERROR: faltan columnas obligatorias en el reporte:")
        for c in faltantes:
            print(f"  - {c}")
        raise SystemExit(1)

    stages = {
        'has_goal':      COL_GOAL in df.columns,
        'has_ontime':    COL_ONTIME in df.columns,
        'has_meet_base': COL_MEET_BASE in df.columns,
        'has_meet_inc':  COL_MEET_INC in df.columns,
        'has_increase':  COL_GROWTH_INC in df.columns,
        'has_reco':      COL_RECO in df.columns,
        'has_name':      'seriesName' in df.columns,
    }

    n = len(df)
    print(f"  Indicadores cargados: {n}")

    if stages['has_goal']:
        con_meta = int(df[COL_GOAL].notna().sum())
        print(f"  Con meta ('{COL_GOAL}'): {con_meta}   |   Sin meta: {n - con_meta}")
        if con_meta == 0:
            print("  [!] Ningun indicador tiene meta: se omiten las donas de convergencia.")
            stages['has_goal'] = False
    else:
        print(f"  [!] Sin columna '{COL_GOAL}': sin punto de meta ni donas de convergencia.")

    if not stages['has_ontime']:
        print(f"  [!] Sin columna '{COL_ONTIME}': el punto de meta se dibujara siempre hueco.")
    if not stages['has_meet_base']:
        print(f"  [!] Sin columna '{COL_MEET_BASE}': convergencia base degradada.")
    if not stages['has_meet_inc']:
        print(f"  [!] Sin columna '{COL_MEET_INC}': convergencia de aumento degradada.")
    if not stages['has_increase']:
        print(f"  [!] Sin columna '{COL_GROWTH_INC}': se omiten TODOS los productos de aumento.")
    if not stages['has_reco']:
        print(f"  [!] Sin columna '{COL_RECO}': se omiten graficas por consideracion y PDF/MD.")
    if not stages['has_name']:
        print("  [i] Sin columna 'seriesName': se usara 'seriesCode' como respaldo.")

    if not stages['has_goal']:
        stages['has_ontime'] = False

    return stages


# ---------------------------------------------------------------------------
# HELPERS DE DONA
# ---------------------------------------------------------------------------
def _pct_text_color(color):
    """Negro sobre gajos claros, blanco sobre gajos oscuros."""
    try:
        r, g, b = to_rgb(color)
    except Exception:
        return 'black'
    return 'black' if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else 'white'


def _draw_donut(ax, counts, labels, colors, legend_fontsize=7, hatches=None):
    """Anillo interior con porcentajes + leyenda. Omite las categorias vacias."""
    keep = [i for i, c in enumerate(counts) if c > 0]
    if not keep:
        return None
    counts_k = [counts[i] for i in keep]
    labels_k = [labels[i] for i in keep]
    colors_k = [colors[i] for i in keep]
    hatches_k = [hatches[i] for i in keep] if hatches else None

    pie, _texts, pcts = ax.pie(
        counts_k, radius=1 - DONUT_WIDTH, startangle=90, counterclock=False,
        colors=colors_k, autopct='%.0f%%', pctdistance=0.79)
    plt.setp(pie, width=DONUT_WIDTH, edgecolor='white')

    for i, p in enumerate(pcts):
        plt.setp(p, color=_pct_text_color(colors_k[i]))

    if hatches_k:
        for wedge, h in zip(pie, hatches_k):
            if h:
                try:
                    wedge.set_hatch(h)
                except Exception:
                    pass

    ax.legend(pie, labels_k, loc="center", bbox_to_anchor=(.25, .5, 0.5, .0),
              fontsize=legend_fontsize, frameon=False)
    return pie


def _arrow(ax, x, y0, y1, color, lw=2.0):
    """Flecha vertical con la punta dimensionada en PUNTOS, no en datos.

    `plt.arrow` mide `head_width` en unidades de datos, asi que en las graficas
    con pocos indicadores (p. ej. las 3 barras de 'escalar') la punta sale
    aplastada y desproporcionada. `annotate` la mide en puntos y se ve igual
    en todos los tamanios de figura.
    """
    if not np.isfinite(y0) or not np.isfinite(y1) or abs(y1 - y0) < 1e-12:
        return
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, linewidth=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=12))


def _y_top(start, end, goal, has_goal, headroom=1.10):
    """Limite superior del eje Y que cubre barras, flechas y puntos de meta."""
    vals = [np.asarray(start, dtype=float), np.asarray(end, dtype=float)]
    g = np.asarray(goal, dtype=float)[np.asarray(has_goal, dtype=bool)]
    if g.size:
        vals.append(g)
    top = np.nanmax(np.concatenate(vals))
    if not np.isfinite(top) or top <= 0:
        top = 1.0
    return top * headroom


def _draw_outer_ring(ax, codes, colors):
    """Anillo exterior de etiquetas (seriesCode) coloreado por indicador."""
    if len(codes) == 0:
        return []
    pie, texts = ax.pie(
        np.ones(len(codes)), radius=1, colors=list(colors), labels=list(codes),
        rotatelabels=True, counterclock=False, startangle=90,
        textprops=dict(va="center", ha='center', rotation_mode='anchor',
                       fontsize=5, color='black'),
        labeldistance=1.17)
    plt.setp(pie, width=DONUT_WIDTH, edgecolor='none')
    return texts


# ---------------------------------------------------------------------------
# 1. BARRAS (replica de graphics_only.plot_baseline / plot_increase)
# ---------------------------------------------------------------------------
def plot_bars(df, stages, escenario):
    """Barras: nivel actual (IF) + flecha al nivel proyectado a N anios.

    A diferencia del original NO se dibuja la flecha intermedia del anio 4:
    ese valor no existe en `final_report_IPP.xlsx`. El cumplimiento en el
    periodo de gobierno se codifica en el relleno del punto de meta.
    """
    growth_col = COL_GROWTH_BASE if escenario == 'baseline' else COL_GROWTH_INC
    print(f"Generando barras del escenario '{escenario}'...")

    start_all, end_all = reconstruct_levels(df, growth_col)
    has_goal, goal, on_time = goal_info(df, stages)

    max_per_plot = 50
    num_plots = int(np.ceil(len(df) / max_per_plot))
    per_plot = int(np.ceil(len(df) / num_plots)) if num_plots > 0 else len(df)

    escritos = []
    for i in range(num_plots):
        s, e = i * per_plot, min((i + 1) * per_plot, len(df))
        label = f'part_{i+1}'
        subset = df.iloc[s:e]
        num_items = len(subset)

        plt.figure(figsize=(12, 4))
        for idx in range(num_items):
            g = s + idx
            row = df.iloc[g]
            plt.bar(idx, start_all[g], color=row.color, width=.65, alpha=.5)
            _arrow(plt.gca(), idx, start_all[g], end_all[g], row.color, lw=2)
            if has_goal[g]:
                if on_time[g]:
                    plt.scatter(idx, goal[g], color='black', s=20, zorder=5)
                else:
                    plt.scatter(idx, goal[g], s=20, facecolors='none',
                                edgecolors='black', linewidths=0.9, zorder=5)

        # `annotate` no participa del autoescalado (a diferencia de `plt.arrow`),
        # asi que el limite superior se fija a mano o las flechas se recortan.
        plt.ylim(0, _y_top(start_all[s:e], end_all[s:e],
                           goal[s:e], has_goal[s:e], headroom=1.14))
        plt.xlim(-1, num_items)
        plt.gca().set_xticks(range(num_items))
        plt.gca().set_xticklabels(subset.seriesCode, rotation=90, fontsize=7)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.ylabel('levels', fontsize=14)
        plt.xlabel('indicators', fontsize=14)

        handles = [
            Line2D([0], [0], color='dimgray', lw=2,
                   label=f'Nivel proyectado a {YEARS_TO_FORECAST} anios'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
                   markeredgecolor='black', markersize=5,
                   label=f'Meta alcanzada en el periodo de gobierno '
                         f'(<= {INTERMEDIATE_CONVERGENCE_YEAR} anios)'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                   markeredgecolor='black', markersize=5,
                   label='Meta no alcanzada a tiempo'),
        ]
        plt.legend(handles=handles, fontsize=6, frameon=False,
                   loc='upper right', ncol=3)

        plt.tight_layout()
        out = f'Bars_{escenario}_{label}.pdf'
        plt.savefig(get_path(out))
        plt.close()
        escritos.append(out)
        print(f"    -> {out} ({num_items} indicadores)")

    return escritos


# ---------------------------------------------------------------------------
# 2. DONA DE PROGRESO (replica exacta: 100 * Crecimiento)
# ---------------------------------------------------------------------------
def plot_donut_progress(df, escenario):
    """Dona de progreso proporcional. Reproduce la formula original sin perdida:
    `100*(row[T_sim-1]-row[0])/row[0]` es identicamente `100*Crecimiento`.
    """
    growth_col = COL_GROWTH_BASE if escenario == 'baseline' else COL_GROWTH_INC
    print(f"Generando dona de progreso del escenario '{escenario}'...")

    start_all, _ = reconstruct_levels(df, growth_col)
    progress = 100.0 * df[growth_col].astype(float).values

    groups = [[], [], [], [], [], []]   # 0: negativo ... 5: > 30%
    for i in range(len(df)):
        if start_all[i] <= 0 or np.isnan(progress[i]):
            continue
        p = progress[i]
        item = (df.iloc[i].seriesCode, df.iloc[i].color)
        if p <= 0:
            groups[0].append(item)
        elif p <= 5:
            groups[1].append(item)
        elif p <= 10:
            groups[2].append(item)
        elif p <= 20:
            groups[3].append(item)
        elif p <= 30:
            groups[4].append(item)
        else:
            groups[5].append(item)

    order = [groups[5], groups[4], groups[3], groups[2], groups[1], groups[0]]
    counts = [len(g) for g in order]
    if sum(counts) == 0:
        print("  [!] Sin datos para la dona de progreso; se omite.")
        return []

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    _draw_donut(
        ax, counts,
        ['mayor a 30%', '20-30%', '10-20%', '5-10%', '0-5%', 'Negativo'],
        ['whitesmoke', 'gainsboro', 'silver', 'darkgray', 'dimgrey', 'black'])

    labels = [c[0] for g in order for c in g]
    cin = [c[1] for g in order for c in g]
    _draw_outer_ring(ax, labels, cin)

    plt.tight_layout()
    out = f'Donut_{escenario}.pdf'
    plt.savefig(get_path(out))
    plt.close()
    print(f"    -> {out}   " + " | ".join(
        f"{l}:{c}" for l, c in zip(
            ['>30%', '20-30%', '10-20%', '5-10%', '0-5%', 'Neg'], counts)))
    return [out]


# ---------------------------------------------------------------------------
# 3. DONA DE CONVERGENCIA - BASELINE
# ---------------------------------------------------------------------------
def plot_donut_convergence_baseline(df, stages):
    """Convergencia reconstruida desde las banderas del reporte.

    APROXIMACION: el gajo 'Tarde' aqui significa "alcanza la meta dentro del
    horizonte completo de N anios" (`Cumple Meta_baseline`), mientras que el
    original usaba `reaches[0]/calibration_index <= YEARS_TO_FORECAST - 1`,
    que excluye los ultimos ~0.8 anios del horizonte. La diferencia solo
    afecta a indicadores que convergen justo al final del periodo.
    """
    print("Generando dona de convergencia del escenario base...")
    if not stages['has_goal']:
        print("  [!] Sin metas disponibles; se omite.")
        return []

    has_goal, _goal, on_time = goal_info(df, stages)

    if stages['has_meet_base']:
        meets = df[COL_MEET_BASE].fillna(0).astype(float).values == 1
    else:
        meets = on_time.copy()
        print(f"  [i] Sin '{COL_MEET_BASE}': 'Tarde' quedara vacio.")

    idx_ontime = [i for i in range(len(df)) if has_goal[i] and on_time[i]]
    idx_late = [i for i in range(len(df)) if has_goal[i] and meets[i] and not on_time[i]]
    idx_unfeas = [i for i in range(len(df)) if has_goal[i] and not meets[i]]
    idx_nogoal = [i for i in range(len(df)) if not has_goal[i]]

    counts = [len(idx_ontime), len(idx_late), len(idx_unfeas), len(idx_nogoal)]
    print(f"  Diagnostico Convergencia (base): A tiempo={counts[0]}, "
          f"Tarde={counts[1]}, Inviable={counts[2]}, Sin meta={counts[3]}")
    if sum(counts) == 0:
        return []

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    _draw_donut(ax, counts, convergence_labels(df, has_goal),
                CONV_COLORS, legend_fontsize=CONV_LEGEND_FONTSIZE,
                hatches=CONV_HATCHES)

    order = idx_ontime + idx_late + idx_unfeas + idx_nogoal
    _draw_outer_ring(ax, [df.iloc[i].seriesCode for i in order],
                     [df.iloc[i].color for i in order])

    plt.tight_layout()
    out = 'Donut_Convergencia_baseline.pdf'
    plt.savefig(get_path(out))
    plt.close()
    print(f"    -> {out}")
    return [out]


# ---------------------------------------------------------------------------
# 4. DONA DE CONVERGENCIA - INCREASE
# ---------------------------------------------------------------------------
def plot_donut_convergence_increase(df, stages):
    """Convergencia del escenario de aumento, con las mismas 4 categorias que
    la base.

    El reporte no trae un 'Cumple_a_tiempo_increase', pero el tiempo SI se
    deduce apoyandose en la monotonia del presupuesto: mas recursos nunca
    desaceleran un indicador (en el peor caso no tienen efecto). De ahi:

    - `Cumple Meta_increase == 0`  -> inviable (no llega dentro del horizonte).
    - Alcanza y ya llegaba A TIEMPO en la base -> a tiempo. Con mas
      presupuesto no puede tardar mas de lo que tardaba con menos.
    - Alcanza y no llegaba a tiempo en la base -> tardia. Cubre dos casos:
      los que ya eran tardios en la base (no pueden empeorar, y adelantarse
      hasta el año objetivo no se puede afirmar con esta fuente) y los
      'jumpers', que en la base no llegaban ni al final del horizonte: que
      salten hasta el año objetivo seria una recuperacion inverosimil.

    Los jumpers se siguen resaltando en verde en el anillo exterior.

    Si falta la bandera de cumplimiento a tiempo de la base, todo lo que
    alcanza cae en 'tardia' (lectura conservadora).
    """
    print("Generando dona de convergencia del escenario de aumento...")
    if not stages['has_goal']:
        print("  [!] Sin metas disponibles; se omite.")
        return []
    if not stages['has_meet_inc']:
        print(f"  [!] Sin columna '{COL_MEET_INC}'; se omite.")
        return []

    has_goal, _goal, on_time_base = goal_info(df, stages)
    meets_inc = df[COL_MEET_INC].fillna(0).astype(float).values == 1
    meets_base = (df[COL_MEET_BASE].fillna(0).astype(float).values == 1
                  if stages['has_meet_base'] else np.zeros(len(df), dtype=bool))

    if not stages['has_ontime']:
        print(f"  [i] Sin '{COL_ONTIME}': todo lo que alcanza se clasifica "
              "como convergencia tardia.")

    idx_ontime, idx_late, idx_unfeas, idx_nogoal = [], [], [], []
    jumpers = []
    for i in range(len(df)):
        if not has_goal[i]:
            idx_nogoal.append(i)
        elif not meets_inc[i]:
            idx_unfeas.append(i)
        elif on_time_base[i]:
            idx_ontime.append(i)            # monotonia: no puede tardar mas
        else:
            idx_late.append(i)
            if stages['has_meet_base'] and not meets_base[i]:
                jumpers.append(i)

    if stages['has_meet_base']:
        print(f"  Jumpers (no alcanzan en base, si con aumento): {len(jumpers)}"
              + (f" -> {[df.iloc[i].seriesCode for i in jumpers]}" if jumpers else "")
              + " -> se clasifican como convergencia tardia.")

    counts = [len(idx_ontime), len(idx_late), len(idx_unfeas), len(idx_nogoal)]
    print(f"  Diagnostico Convergencia (aumento): A tiempo={counts[0]}, "
          f"Tarde={counts[1]}, Inviable={counts[2]}, Sin meta={counts[3]}")
    if sum(counts) == 0:
        return []

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    _draw_donut(ax, counts, convergence_labels(df, has_goal),
                CONV_COLORS, legend_fontsize=CONV_LEGEND_FONTSIZE,
                hatches=CONV_HATCHES)

    order = idx_ontime + idx_late + idx_unfeas + idx_nogoal
    texts = _draw_outer_ring(ax, [df.iloc[i].seriesCode for i in order],
                             [df.iloc[i].color for i in order])
    for i, orig in enumerate(order):
        if orig in jumpers and i < len(texts):
            texts[i].set_color('green')
            texts[i].set_weight('bold')
            texts[i].set_fontsize(5)

    # Nota al pie, no titulo: a radio 1.17 las etiquetas del anillo exterior
    # invaden la zona del titulo.
    fig.text(0.5, 0.015,
             'Convergencia con aumento presupuestal. El tiempo se deriva del '
             'escenario base asumiendo que mas presupuesto nunca desacelera un\n'
             'indicador: quien llega a tiempo en la base llega a tiempo aqui; '
             'el resto de los que alcanzan la meta se clasifica como tardio.\n'
             'En verde, los indicadores que no alcanzaban la meta en la base y '
             'si con el aumento.',
             ha='center', va='bottom', fontsize=6)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = 'Donut_Convergencia_increase.pdf'
    plt.savefig(get_path(out))
    plt.close()
    print(f"    -> {out}")
    return [out]


# ---------------------------------------------------------------------------
# 5. BARRAS POR CONSIDERACION
# ---------------------------------------------------------------------------
def plot_by_consideration(df, stages):
    """Reimplementacion local de prospective_simulation_byconsideration.py.

    No se puede reutilizar la original porque lee `output_baseline.xlsx` y
    `raw_indicators.xlsx`. Usa `Recomendacion_Final` TAL CUAL viene en el
    reporte (no se recalcula).
    """
    print("\n" + "=" * 50)
    print("GENERANDO GRAFICAS POR RECOMENDACION")
    print("=" * 50)

    if not stages['has_reco']:
        print(f"  [!] Sin columna '{COL_RECO}'; se omite esta etapa.")
        return []

    start_all, end_all = reconstruct_levels(df, COL_GROWTH_BASE)
    has_goal, goal, on_time = goal_info(df, stages)

    escritos = []
    for reco in RECOMMENDATIONS:
        pos = [i for i in range(len(df)) if df.iloc[i][COL_RECO] == reco]
        if not pos:
            print(f"[-] Sin indicadores para: '{reco}'... omitiendo.")
            continue

        if len(pos) > 50:
            mid = len(pos) // 2
            groups = [(pos[:mid], f"{FILE_SUFFIXES[reco]}_1"),
                      (pos[mid:], f"{FILE_SUFFIXES[reco]}_2")]
            print(f"[+] Dividiendo {len(pos)} indicadores para: '{reco}' en dos partes.")
        else:
            groups = [(pos, FILE_SUFFIXES[reco])]
            print(f"[+] Graficando {len(pos)} indicadores para: '{reco}'")

        for idx_list, suffix in groups:
            num_items = len(idx_list)
            fig_width = 12 if num_items >= 30 else max(8, num_items * 0.4)
            plt.figure(figsize=(fig_width, 4))

            for x_pos, g in enumerate(idx_list):
                row = df.iloc[g]
                plt.bar(x_pos, start_all[g], color=row.color, width=.65, alpha=.4)
                _arrow(plt.gca(), x_pos, start_all[g], end_all[g], row.color, lw=2.5)
                if has_goal[g]:
                    if on_time[g]:
                        plt.scatter(x_pos, goal[g], color='black', s=25, zorder=10)
                    else:
                        plt.scatter(x_pos, goal[g], s=25, facecolors='none',
                                    edgecolors='black', linewidths=0.9, zorder=10)

            sel = np.array(idx_list)
            plt.ylim(0, _y_top(start_all[sel], end_all[sel],
                               goal[sel], has_goal[sel], headroom=1.08))
            plt.xlim(-1, num_items)
            plt.xticks(range(num_items),
                       [df.iloc[g].seriesCode for g in idx_list],
                       rotation=90, fontsize=7)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.ylabel('levels', fontsize=12)
            plt.xlabel('indicators', fontsize=12)
            plt.tight_layout()

            out = f'Bars_baseline_by_consideration_{suffix}.pdf'
            plt.savefig(get_path(out))
            plt.close()
            escritos.append(out)
            print(f"    -> Guardada en: {out}")

    return escritos


# ---------------------------------------------------------------------------
# 6. PDF + MARKDOWN DE RECOMENDACIONES (reutiliza final_report_generator)
# ---------------------------------------------------------------------------
def generate_summary_documents(df, stages):
    """Reutiliza generate_visual_table() y generate_markdown_report().

    IMPORTANTE: nunca se llama a generate_report(), porque su linea 167 hace
    `df_final.to_excel(file_report)` sobre Outputs/final_report_IPP.xlsx y
    sobrescribiria el archivo curado que aqui es la ENTRADA.
    """
    print("\nGenerando PDF y Markdown de recomendaciones...")
    if not stages['has_reco']:
        print(f"  [!] Sin columna '{COL_RECO}'; se omite esta etapa.")
        return []

    name_col = 'seriesName' if stages['has_name'] else 'seriesCode'
    buckets = {r: df.loc[df[COL_RECO] == r, name_col].tolist()
               for r in RECOMMENDATIONS}
    green = buckets["Continuar programas"]
    yellow = buckets["Escalar programas"]
    red = buckets["Revisar los programas asociados"]
    print(f"  Continuar={len(green)}, Escalar={len(yellow)}, Revisar={len(red)}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import final_report_generator as frg

    _pdf, _md = frg.file_pdf, frg.file_md
    frg.file_pdf = get_path('Resumen_Recomendaciones_IPP.pdf')
    frg.file_md = get_path('Resumen_Recomendaciones_IPP.md')
    escritos = []
    try:
        try:
            frg.generate_visual_table(green, yellow, red)
            escritos.append('Resumen_Recomendaciones_IPP.pdf')
        except Exception as e:
            print(f"Error generando PDF: {e}")
        try:
            frg.generate_markdown_report(green, yellow, red)
            escritos.append('Resumen_Recomendaciones_IPP.md')
        except Exception as e:
            print(f"Error generando MD: {e}")
    finally:
        frg.file_pdf, frg.file_md = _pdf, _md

    return escritos


# ---------------------------------------------------------------------------
# 7. INFORME DE CONSISTENCIA
# ---------------------------------------------------------------------------
def write_consistency_report(df, stages):
    """Reporta (sin corregir) las incoherencias internas del Excel fusionado."""
    print("\nGenerando informe de consistencia...")
    L = ["# Informe de consistencia - `final_report_IPP.xlsx`", "",
         "Este informe **reporta**, no corrige. El script de graficas respeta "
         "las columnas del Excel tal como vienen.", "",
         f"- Indicadores analizados: **{len(df)}**"]

    year_cols = get_year_columns(df)
    if year_cols:
        last_per_row = []
        for _, row in df.iterrows():
            yrs = [y for y, c in year_cols if pd.notna(row[c])]
            last_per_row.append(max(yrs) if yrs else None)
        df_tmp = pd.DataFrame({'seriesCode': df['seriesCode'].values,
                               'ultimo_anio': last_per_row})
        L += ["", "## Cobertura temporal por corrida", "",
              "| Ultimo anio observado | n | Indicadores |", "|---|---|---|"]
        for yr, sub in df_tmp.groupby('ultimo_anio', dropna=False):
            codes = ", ".join(f"`{c}`" for c in sub['seriesCode'])
            L.append(f"| {yr} | {len(sub)} | {codes} |")

    if stages['has_goal']:
        sin_meta = df.loc[df[COL_GOAL].isna(), 'seriesCode'].tolist()
        L += ["", f"## Indicadores sin meta ({len(sin_meta)})", ""]
        L += ([f"- `{c}`" for c in sin_meta] if sin_meta
              else ["*Todos los indicadores tienen meta.*"])

    # Ultima_milla: 1 si IF > 0.9  (final_report_generator.py:140)
    if 'Ultima_milla' in df.columns:
        calc = (df['IF'].astype(float) > 0.9).astype(int)
        bad = df.index[calc.values != df['Ultima_milla'].fillna(-1).astype(int).values]
        L += ["", f"## `Ultima_milla` que no se deriva de `IF > 0.9` ({len(bad)})", ""]
        if len(bad):
            L += ["| seriesCode | IF | archivo | recalculado |", "|---|---|---|---|"]
            for i in bad:
                L.append(f"| `{df.loc[i,'seriesCode']}` | {df.loc[i,'IF']:.4f} | "
                         f"{int(df.loc[i,'Ultima_milla'])} | {int(calc.loc[i])} |")
        else:
            L.append("*Sin discrepancias.*")

    # Elastico: 1 si (Crec_inc - Crec_base) > 0.03  (final_report_generator.py:131)
    if 'Elastico' in df.columns and stages['has_increase']:
        diff = df[COL_GROWTH_INC].astype(float) - df[COL_GROWTH_BASE].astype(float)
        calc = (diff > 0.03).astype(int)
        bad = df.index[calc.values != df['Elastico'].fillna(-1).astype(int).values]
        L += ["", f"## `Elastico` que no se deriva de `delta crecimiento > 0.03` ({len(bad)})", ""]
        if len(bad):
            L += ["| seriesCode | delta crecimiento | archivo | recalculado |", "|---|---|---|---|"]
            for i in bad:
                L.append(f"| `{df.loc[i,'seriesCode']}` | {diff.loc[i]:.4f} | "
                         f"{int(df.loc[i,'Elastico'])} | {int(calc.loc[i])} |")
        else:
            L.append("*Sin discrepancias.*")

    # Recomendacion_Final: arbol de final_report_generator.py:142-150
    if (stages['has_reco'] and stages['has_ontime']
            and 'Ultima_milla' in df.columns and 'Elastico' in df.columns
            and stages['has_meet_inc']):
        ct = df[COL_ONTIME].fillna(0).astype(float)
        um = df['Ultima_milla'].fillna(0).astype(float)
        el = df['Elastico'].fillna(0).astype(float)
        cmi = df[COL_MEET_INC].fillna(0).astype(float)
        calc = np.where((ct == 1) | (um == 1), "Continuar programas",
                        np.where((el == 1) & (cmi == 1), "Escalar programas",
                                 "Revisar los programas asociados"))
        bad = df.index[calc != df[COL_RECO].values]
        L += ["", f"## `{COL_RECO}` que no se deriva de las banderas ({len(bad)})", ""]
        if len(bad):
            L += ["| seriesCode | en el archivo | recalculado | a_tiempo | ult_milla | elastico | meta_aumento |",
                  "|---|---|---|---|---|---|---|"]
            for i in bad:
                L.append(f"| `{df.loc[i,'seriesCode']}` | {df.loc[i,COL_RECO]} | "
                         f"{calc[df.index.get_loc(i)]} | {ct.loc[i]:.0f} | "
                         f"{um.loc[i]:.0f} | {el.loc[i]:.0f} | {cmi.loc[i]:.0f} |")
            L += ["", "> Esperable al fusionar dos corridas distintas. "
                  "El script de graficas usa la columna del archivo, no el recalculo."]
        else:
            L.append("*Sin discrepancias.*")

    L.append("")
    out = 'reporte_consistencia.md'
    with open(get_path(out), 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    print(f"    -> {out}")
    return [out]


# ---------------------------------------------------------------------------
# ORQUESTADOR
# ---------------------------------------------------------------------------
def _stage(nombre, fn, escritos, omitidos, *args):
    try:
        escritos.extend(fn(*args) or [])
    except Exception as e:
        omitidos.append(f"{nombre}: {type(e).__name__}: {e}")
        print(f"  [X] Etapa '{nombre}' omitida por error: {e}")


def generate_graphics_from_report(input_file=None):
    print("\n" + "=" * 50)
    print("GRAFICAS A PARTIR DE final_report_IPP.xlsx")
    print("=" * 50)

    file_report = input_file or get_input_path(INPUT_FILENAME)
    if not os.path.exists(file_report):
        print(f"ERROR: No se encontro el reporte de entrada:\n - {file_report}")
        raise SystemExit(1)
    print(f"Entrada : {file_report}")
    print(f"Salida  : {os.path.join('Outputs', OUTPUT_SUBDIR)}")

    df = pd.read_excel(file_report)
    stages = check_availability(df)

    selected_sdgs = load_selected_sdgs()
    print(f"\nODS seleccionados: {selected_sdgs}")
    df = filter_by_sdg(df, selected_sdgs)
    df = df.reset_index(drop=True)
    if len(df) == 0:
        print("Advertencia: ningun indicador coincide con los ODS seleccionados.")
        return
    print(f"Indicadores a graficar tras el filtro: {len(df)}\n")

    escritos, omitidos = [], []

    _stage('Bars baseline', plot_bars, escritos, omitidos, df, stages, 'baseline')
    _stage('Donut baseline', plot_donut_progress, escritos, omitidos, df, 'baseline')
    _stage('Donut convergencia baseline', plot_donut_convergence_baseline,
           escritos, omitidos, df, stages)

    if stages['has_increase']:
        _stage('Bars increase', plot_bars, escritos, omitidos, df, stages, 'increase')
        _stage('Donut increase', plot_donut_progress, escritos, omitidos, df, 'increase')
        _stage('Donut convergencia increase', plot_donut_convergence_increase,
               escritos, omitidos, df, stages)
    else:
        omitidos.append(f"Escenario de aumento: falta la columna '{COL_GROWTH_INC}'.")

    _stage('Barras por consideracion', plot_by_consideration, escritos, omitidos, df, stages)
    _stage('PDF/MD de recomendaciones', generate_summary_documents, escritos, omitidos, df, stages)
    _stage('Informe de consistencia', write_consistency_report, escritos, omitidos, df, stages)

    print("\n" + "=" * 50)
    print(f"RESUMEN - {len(escritos)} archivo(s) en Outputs/{OUTPUT_SUBDIR}/")
    print("=" * 50)
    for f in escritos:
        print(f"  [OK] {f}")
    if omitidos:
        print("\nEtapas omitidas:")
        for m in omitidos:
            print(f"  [--] {m}")
    print("\nGeneracion completada.")


if __name__ == "__main__":
    generate_graphics_from_report(sys.argv[1] if len(sys.argv) > 1 else None)
