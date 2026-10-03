"""Metas oficiales: nunca inferirlas de las metas técnicas `goals` / `goal`."""

import os
import numpy as np
import pandas as pd


NO_GOAL_MESSAGE = "Convergencia no evaluable: no hay metas oficiales disponibles"
NO_GOAL_GUIDANCE = (
    "Sin meta oficial no se evalúan cumplimiento ni convergencia. "
    "Interprete las trayectorias, niveles, crecimiento, sensibilidad presupuestaria "
    "y última milla; la meta técnica de simulación no es una meta gubernamental."
)

# Estilo compartido por las donas y el pie del resumen PDF (tamaño en puntos).
CONVERGENCE_NOTE_FONT = 'DejaVu Sans'
CONVERGENCE_NOTE_SIZE = 6
CONVERGENCE_NOTE_COLOR = 'dimgray'
CONVERGENCE_NOTE_GAP_INCHES = 0.16


def convergence_note(with_goal, without_goal):
    return f'Con meta oficial: {with_goal} | Sin meta oficial (excluidos): {without_goal}'


def read_raw_indicators(get_path):
    """Preserva la lectura histórica; solo gov_target usa parsing estricto."""
    path = get_path('raw_indicators.xlsx')
    data = pd.read_excel(path)
    if 'gov_target' in data:
        targets = pd.read_excel(path, usecols=['gov_target'], keep_default_na=False,
                                na_values=[''], dtype={'gov_target': object})
        data['gov_target'] = targets['gov_target']
    return data


def normalize_official_goals(data):
    """Vacío = ausencia; texto erróneo, infinito o fuera de (0, 1) = error.

    Cero es un valor ingresado: se normaliza y valida con los mismos bounds.
    Use keep_default_na=False al leer el Excel para no ocultar textos como 'NA'.
    """
    targets = data.get('gov_target', pd.Series(np.nan, index=data.index))
    normalized = pd.Series(np.nan, index=data.index, dtype=float)
    errors = []
    for index, value in targets.items():
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            continue
        code = data.loc[index, 'seriesCode']
        try:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("no es un número")
            numeric = float(value)
            denominator = data.loc[index, 'bestbound'] - data.loc[index, 'worstbound']
            goal = (numeric - data.loc[index, 'worstbound']) / denominator
            if not np.isfinite(numeric) or not np.isfinite(goal) or not 0 < goal < 1:
                raise ValueError("debe normalizarse dentro de (0, 1)")
            normalized.loc[index] = goal
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f" - {code}: gov_target={value!r} ({exc})")
    if errors:
        raise ValueError("Metas oficiales inválidas en gov_target:\n" + "\n".join(errors))
    return normalized, normalized.notna()


def official_goal_data(data, *sources):
    """Resuelve evidencia real por seriesCode, también para resultados antiguos.

    La primera fuente con real_goal/real_goals (incluso vacío) es autoritativa.
    Si faltan esas columnas, se consulta la preparación o el gov_target original.
    Una bandera falsa excluye la meta. Una bandera verdadera sin valor real nunca
    basta para evaluar. No se consultan `goal` ni `goals`.
    """
    codes = data['seriesCode']
    real = pd.Series(np.nan, index=data.index, dtype=float)
    resolved = pd.Series(False, index=data.index)
    for source in (data,) + sources:
        if source is None:
            continue
        column = next((c for c in ('real_goal', 'real_goals') if c in source), None)
        if column is not None:
            values = pd.to_numeric(source[column], errors='raise').astype(float)
            invalid = values.notna() & (~np.isfinite(values) | (values <= 0) | (values >= 1))
            if invalid.any():
                raise ValueError(f"Metas oficiales normalizadas inválidas: {source.loc[invalid, 'seriesCode'].tolist()}")
        elif 'gov_target' in source:
            values, _ = normalize_official_goals(source)
        else:
            continue
        if 'has_official_goal' in source:
            # Acepta los booleanos de Excel y su representación textual.
            flag = source['has_official_goal'].map(
                lambda v: str(v).strip().lower() in ('true', '1', '1.0') if pd.notna(v) else None
            )
            values = values.mask(flag == False)
        lookup = pd.Series(values.values, index=source['seriesCode'])
        available = ~resolved & codes.isin(lookup.index)
        real.loc[available] = codes.loc[available].map(lookup)
        resolved.loc[available] = True
    return real, real.notna()


def attach_official_goals(data, *sources):
    result = data.copy()
    result['real_goal'], result['has_official_goal'] = official_goal_data(data, *sources)
    return result


def load_goal_sources(get_path):
    sources = []
    for filename in ('data_indicators.xlsx', 'raw_indicators.xlsx'):
        path = get_path(filename)
        if os.path.exists(path):
            sources.append(read_raw_indicators(get_path) if filename.startswith('raw_') else pd.read_excel(path))
    return sources
