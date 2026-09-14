"""Невязка между расчётом (симулятор) и фактом (история).

Факт берётся из `eclipse_schedule.diagnostics.field_rates()` - таблица
с колонками `date` и дебитами (`liquid_rate`, `oil_rate`, `water_rate`,
...). "Расчёт" ожидается в том же формате: пользователь получает его,
прогнав реальный симулятор на кейсе и прочитав UNSMRY (например через
`resdata`/`ecl2df`) - конвертация UNSMRY в такую таблицу в этот пакет
не входит, так как без реального прогона её нечем проверить.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_VALUE_COLS = ("liquid_rate", "water_rate")


def misfit(
    observed: pd.DataFrame,
    simulated: pd.DataFrame,
    value_cols: tuple[str, ...] = DEFAULT_VALUE_COLS,
) -> float:
    """NRMSE (RMSE, нормированная на RMS факта), усреднённая по `value_cols`.

    Меньше - лучше. Сравнение идёт только по датам, общим для `observed`
    и `simulated` (INNER JOIN по `date`).
    """
    merged = pd.merge(
        observed[["date", *value_cols]], simulated[["date", *value_cols]], on="date", suffixes=("_obs", "_sim")
    )
    if merged.empty:
        raise ValueError("Нет общих дат между observed и simulated")

    total = 0.0
    for col in value_cols:
        obs = merged[f"{col}_obs"].to_numpy(dtype=float)
        sim = merged[f"{col}_sim"].to_numpy(dtype=float)
        denom = np.sqrt(np.mean(obs**2))
        rmse = np.sqrt(np.mean((obs - sim) ** 2))
        total += rmse / denom if denom > 0 else rmse
    return total / len(value_cols)


def ensemble_misfits(
    observed: pd.DataFrame,
    simulated_by_case: dict[int, pd.DataFrame],
    value_cols: tuple[str, ...] = DEFAULT_VALUE_COLS,
) -> pd.Series:
    """misfit() для каждого кейса ансамбля - Series {case_id: misfit}, отсортирована по возрастанию."""
    return pd.Series(
        {case_id: misfit(observed, df, value_cols) for case_id, df in simulated_by_case.items()}
    ).sort_values()


def observation_vector(
    observed: pd.DataFrame,
    dates: list[pd.Timestamp],
    value_cols: tuple[str, ...] = DEFAULT_VALUE_COLS,
    obs_std_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Факт в виде плоского вектора для ES-MDA: сначала все даты по первой
    колонке из `value_cols`, потом по второй и т.д. (тот же порядок должен
    использовать :func:`predictions_matrix`). Возвращает (observed, std) -
    std берётся как доля `obs_std_frac` от модуля значения (с нижним
    порогом, чтобы не получить нулевую дисперсию там, где факт точно 0).
    """
    indexed = observed.set_index("date")
    parts = []
    for col in value_cols:
        series = indexed[col].reindex(dates)
        if series.isna().any():
            raise ValueError(f"В observed нет значений {col!r} на все даты из `dates`")
        parts.append(series.to_numpy(dtype=float))
    obs_vector = np.concatenate(parts)

    std_vector = np.abs(obs_vector) * obs_std_frac
    floor = max(np.abs(obs_vector).mean() * obs_std_frac, 1e-6)
    std_vector = np.where(std_vector <= 0, floor, std_vector)
    return obs_vector, std_vector


def predictions_matrix(
    simulated: list[pd.DataFrame],
    dates: list[pd.Timestamp],
    value_cols: tuple[str, ...] = DEFAULT_VALUE_COLS,
) -> np.ndarray:
    """Расчёты всего ансамбля в виде матрицы (n_obs, n_ens) - тот же порядок
    строк, что и :func:`observation_vector`, один столбец на реализацию.
    """
    columns = []
    for sim in simulated:
        indexed = sim.set_index("date")
        parts = []
        for col in value_cols:
            series = indexed[col].reindex(dates)
            if series.isna().any():
                raise ValueError(f"В одной из реализаций нет значений {col!r} на все даты из `dates`")
            parts.append(series.to_numpy(dtype=float))
        columns.append(np.concatenate(parts))
    return np.column_stack(columns)
