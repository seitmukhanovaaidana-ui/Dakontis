"""Контроль качества истории (WCONHIST/WCONINJH) перед адаптацией модели.

Реальные измеренные дебиты у разных скважин в один и тот же месяц
практически никогда не совпадают до 5-6 значащих цифр. Если такое
совпадение есть - это почти всегда признак того, что вместо факта в
экспорт попало какое-то дефолтное/заглушечное значение (например,
минимальный ненулевой дебит вместо пропуска), и такую точку не стоит
использовать как цель адаптации без проверки по первичным данным.
"""

from __future__ import annotations

import pandas as pd

_RATE_COLUMNS = ("oil_rate", "water_rate", "gas_rate")


def detect_placeholder_rates(history: pd.DataFrame, min_wells: int = 2) -> pd.DataFrame:
    """Ищет подозрительные дебиты: одно и то же ненулевое значение делят
    между собой `min_wells` и более разных скважин на одну и ту же дату.

    Возвращает таблицу: date, field, value, n_wells, wells - отсортированную
    по убыванию n_wells. Пустая таблица с теми же столбцами, если ничего
    не найдено.
    """
    columns = ["date", "field", "value", "n_wells", "wells"]
    producers = history[history["kind"] == "producer"]
    if producers.empty:
        return pd.DataFrame(columns=columns)

    suspects = []
    for field in _RATE_COLUMNS:
        if field not in producers.columns:
            continue
        positive = producers[producers[field] > 0]
        grouped = positive.groupby(["date", field])["well"].apply(list)
        for (date, value), wells in grouped.items():
            if len(wells) >= min_wells:
                suspects.append({
                    "date": date,
                    "field": field,
                    "value": value,
                    "n_wells": len(wells),
                    "wells": wells,
                })

    if not suspects:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(suspects, columns=columns)
        .sort_values(["n_wells", "date"], ascending=[False, True])
        .reset_index(drop=True)
    )
