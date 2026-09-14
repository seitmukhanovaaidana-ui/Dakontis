"""Диагностика истории напрямую из SCHEDULE, без запуска симулятора.

`history` (из :func:`eclipse_schedule.parser.parse_schedule`) уже содержит
факт по скважинам помесячно - этого достаточно, чтобы пересчитать
дебиты по полю (для сверки с графиком типа "Field Liquid production
rate") и оценить баланс закачка/отбор (VRR) ещё до тяжёлого 3D-прогона.
"""

from __future__ import annotations

import pandas as pd


def field_rates(history: pd.DataFrame) -> pd.DataFrame:
    """Суммирует дебиты/закачку по всем скважинам на каждую дату.

    Возвращает таблицу с колонками: date, oil_rate, water_rate, gas_rate,
    liquid_rate, active_producers, injection_rate, active_injectors.
    """
    producers = history[history["kind"] == "producer"]
    if producers.empty:
        raise ValueError("В history нет строк с kind == 'producer'")

    field = producers.groupby("date").agg(
        oil_rate=("oil_rate", "sum"),
        water_rate=("water_rate", "sum"),
        gas_rate=("gas_rate", "sum"),
        active_producers=("well", "nunique"),
    )
    field["liquid_rate"] = field["oil_rate"] + field["water_rate"]

    injectors = history[history["kind"] == "injector"]
    if not injectors.empty:
        inj = injectors.groupby("date").agg(
            injection_rate=("surface_rate", "sum"),
            active_injectors=("well", "nunique"),
        )
        field = field.join(inj, how="left")

    for col in ("injection_rate", "active_injectors"):
        if col not in field.columns:
            field[col] = 0.0
    field[["injection_rate", "active_injectors"]] = field[["injection_rate", "active_injectors"]].fillna(0)

    return field.reset_index().sort_values("date").reset_index(drop=True)


def voidage_replacement_ratio(field_rates_df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Добавляет к таблице field_rates() отношение закачки к отбору жидкости (VRR).

    `window` - если задан, дополнительно считается скользящее среднее VRR
    по стольким последним периодам (сглаживает помесячный шум).
    """
    df = field_rates_df.copy()
    liquid = df["liquid_rate"].replace(0, pd.NA)
    df["vrr"] = df["injection_rate"] / liquid
    if window:
        df["vrr_rolling"] = df["vrr"].rolling(window, min_periods=1).mean()
    return df
