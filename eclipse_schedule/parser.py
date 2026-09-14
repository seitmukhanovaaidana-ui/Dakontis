"""Разбор SCHEDULE-секции Eclipse/Petrel (WELSPECS, COMPDAT, DATES,
WCONHIST, WCONINJH) в таблицы pandas.

Симулятор здесь не запускается и не нужен - файл вроде `FINAL_SCH.INC`
уже содержит саму историю (дебиты/закачку по скважинам помесячно),
которая используется как факт при адаптации модели (history matching).

Разбирается не полная грамматика Eclipse, а только блоки, нужные для
контроля качества и диагностики истории: остальные ключевые слова
(TUNING, WLIST, GRUPTREE, RPTSCHED, RPTRST, FBHPDEF, SKIP/ENDSKIP)
пропускаются как есть, без интерпретации.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Ключевые слова, которые распознаются как начало нового блока. Список не
# обязан быть полным (в SCH.INC могут встретиться и другие) - все, что не
# входит в этот набор, просто останется частью "данных" текущего блока,
# что безопасно, поскольку интересующие нас блоки (WELSPECS/COMPDAT/DATES/
# WCONHIST/WCONINJH) всегда корректно завершаются одиночным "/".
_KNOWN_KEYWORDS = {
    "WELSPECS", "COMPDAT", "DATES", "WCONHIST", "WCONINJH",
    "WLIST", "GRUPTREE", "TUNING", "RPTSCHED", "RPTRST",
    "SKIP", "ENDSKIP", "FBHPDEF", "WCONPROD", "WEFAC", "WELOPEN",
}

_TOKEN_RE = re.compile(r"'[^']*'|\S+")


def _strip_comment(line: str) -> str:
    """Убирает хвост '-- ...', не трогая '--' внутри кавычек."""
    in_quote = False
    for i, ch in enumerate(line[:-1]):
        if ch == "'":
            in_quote = not in_quote
        elif not in_quote and ch == "-" and line[i + 1] == "-":
            return line[:i]
    return line


def _read_lines(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def iter_keyword_blocks(lines: list[str]):
    """Выдаёт (keyword, data_lines, start_idx, end_idx) для каждого блока.

    Блок идёт от строки-ключевого слова до следующей строки-ключевого
    слова или строки-одиночного "/" (что раньше встретится). Индексы -
    0-based номера строк в исходном списке `lines`, `end_idx` не
    включается (как в срезах).
    """
    i = 0
    n = len(lines)
    while i < n:
        raw = _strip_comment(lines[i]).strip()
        if raw in _KNOWN_KEYWORDS:
            keyword = raw
            start = i
            i += 1
            data: list[str] = []
            while i < n:
                candidate = _strip_comment(lines[i]).strip()
                if candidate == "/":
                    i += 1
                    break
                if candidate in _KNOWN_KEYWORDS:
                    break
                if candidate:
                    data.append(candidate)
                i += 1
            yield keyword, data, start, i
        else:
            i += 1


def _tokenize(line: str) -> list[str]:
    tokens = _TOKEN_RE.findall(line)
    return [t[1:-1] if t.startswith("'") and t.endswith("'") else t for t in tokens if t != "/"]


def _to_float(token: str | None) -> float | None:
    if token is None:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _to_int(token: str | None) -> int | None:
    value = _to_float(token)
    return int(value) if value is not None else None


def _parse_date(line: str) -> pd.Timestamp:
    tokens = _tokenize(line)
    day, month, year = tokens[0], tokens[1].upper(), tokens[2]
    return pd.Timestamp(year=int(year), month=_MONTHS[month], day=int(day))


@dataclass
class ScheduleData:
    """Результат разбора файла SCHEDULE."""

    wells: pd.DataFrame          # well, group, i, j, phase, date (дата ввода)
    completions: pd.DataFrame    # well, i, j, k1, k2, status, conn_factor, date
    history: pd.DataFrame        # well, date, kind, status, oil/water/gas rate или surface_rate
    dates: list[pd.Timestamp]    # все даты из блоков DATES, по порядку


def parse_schedule(path: str | Path, start_date: str | pd.Timestamp | None = None) -> ScheduleData:
    """Разбирает файл SCHEDULE (например `FINAL_SCH.INC`) в таблицы pandas.

    `start_date` - дата START из основного .DATA файла (например
    '1973-06-01'). Она нужна, чтобы правильно датировать самые первые
    WELSPECS/COMPDAT/WCONHIST блоки, которые в файле идут ДО первого
    DATES (они описывают состояние на момент START). Если не задать,
    у таких строк будет `date = NaT`.
    """
    lines = _read_lines(path)
    current_date = pd.Timestamp(start_date) if start_date is not None else pd.NaT

    well_rows: list[dict] = []
    compdat_rows: list[dict] = []
    history_rows: list[dict] = []
    dates: list[pd.Timestamp] = []

    for keyword, data, _start, _end in iter_keyword_blocks(lines):
        if keyword == "DATES":
            for line in data:
                current_date = _parse_date(line)
                dates.append(current_date)

        elif keyword == "WELSPECS":
            for line in data:
                toks = _tokenize(line)
                if len(toks) < 4:
                    continue
                well_rows.append({
                    "well": toks[0],
                    "group": toks[1],
                    "i": _to_int(toks[2]),
                    "j": _to_int(toks[3]),
                    "phase": toks[5] if len(toks) > 5 else None,
                    "date": current_date,
                })

        elif keyword == "COMPDAT":
            for line in data:
                toks = _tokenize(line)
                if len(toks) < 6:
                    continue
                compdat_rows.append({
                    "well": toks[0],
                    "i": _to_int(toks[1]),
                    "j": _to_int(toks[2]),
                    "k1": _to_int(toks[3]),
                    "k2": _to_int(toks[4]),
                    "status": toks[5],
                    "conn_factor": _to_float(toks[7]) if len(toks) > 7 else None,
                    "date": current_date,
                })

        elif keyword == "WCONHIST":
            for line in data:
                toks = _tokenize(line)
                if len(toks) < 6:
                    continue
                history_rows.append({
                    "well": toks[0],
                    "date": current_date,
                    "kind": "producer",
                    "status": toks[1],
                    "cmode": toks[2],
                    "oil_rate": _to_float(toks[3]) or 0.0,
                    "water_rate": _to_float(toks[4]) or 0.0,
                    "gas_rate": _to_float(toks[5]) or 0.0,
                })

        elif keyword == "WCONINJH":
            for line in data:
                toks = _tokenize(line)
                if len(toks) < 4:
                    continue
                history_rows.append({
                    "well": toks[0],
                    "date": current_date,
                    "kind": "injector",
                    "status": toks[2],
                    "inj_type": toks[1],
                    "surface_rate": _to_float(toks[3]) or 0.0,
                })

    wells = pd.DataFrame(well_rows)
    completions = pd.DataFrame(compdat_rows)
    history = pd.DataFrame(history_rows)
    return ScheduleData(wells=wells, completions=completions, history=history, dates=dates)
