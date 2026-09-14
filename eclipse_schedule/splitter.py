"""Обрезка файла SCHEDULE по дате - для blind-test проверки адаптации.

Стандартный приём: адаптировать модель только на части истории (скажем,
до 2015 года), а оставшийся хвост использовать как "слепой" тест -
сравнить прогноз модели с фактом, которого модель не видела при
адаптации. Функция ниже готовит такой обрезанный `*_SCH.INC`, вырезая
всё начиная с первого блока DATES, который наступает позже cutoff_date
(вместе со всеми относящимися к нему WELSPECS/COMPDAT/WCONHIST/WCONINJH) -
и сохраняя исходное форматирование и комментарии для всего, что остаётся.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .parser import _parse_date, _read_lines, iter_keyword_blocks


def truncate_schedule(
    input_path: str | Path,
    output_path: str | Path,
    cutoff_date: str | pd.Timestamp,
) -> pd.Timestamp | None:
    """Обрезает `input_path` по `cutoff_date` (включительно) и пишет в `output_path`.

    Возвращает дату последнего блока DATES, оставшегося в файле (или None,
    если в файле не было ни одного DATES раньше или равного cutoff_date -
    тогда сохраняется исходный файл целиком).
    """
    cutoff = pd.Timestamp(cutoff_date)
    lines = _read_lines(input_path)

    keep_until = len(lines)
    last_kept_date: pd.Timestamp | None = None

    for keyword, data, start, _end in iter_keyword_blocks(lines):
        if keyword != "DATES":
            continue
        block_dates = [_parse_date(line) for line in data]
        if any(d > cutoff for d in block_dates):
            keep_until = start
            break
        last_kept_date = block_dates[-1]

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines[:keep_until]) + "\n", encoding="utf-8")
    return last_kept_date
