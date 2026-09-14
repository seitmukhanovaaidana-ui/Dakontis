"""Чтение результатов реального прогона (UNSMRY) в формат `field_rates`.

Это единственное место во всём `history_match`, где нужен реальный
прогон симулятора и реальный файл на диске - поэтому единственное,
что нельзя проверить без Eclipse/tNavigator/OPM Flow. Сам разбор
UNSMRY делает библиотека `resdata` (Equinor, тот же движок, что
использует ERT) - здесь только маппинг её ключей (`FLPR`, `FOPR`, ...)
на колонки, которые ожидают `history_match.objective` и
`eclipse_schedule.diagnostics.field_rates()` (`liquid_rate`,
`oil_rate`, ...).

`resdata` - необязательная зависимость (см. requirements.txt): она
нужна только для этого модуля, весь остальной `history_match` работает
без неё. Импортируется лениво (внутри функции), поэтому её отсутствие
не ломает остальной пакет.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_KEY_MAP = {
    "FLPR": "liquid_rate",
    "FOPR": "oil_rate",
    "FWPR": "water_rate",
    "FGPR": "gas_rate",
    "FWIR": "injection_rate",
}


def read_field_rates(
    data_path: str | Path,
    key_map: dict[str, str] | None = None,
    case_root: str | Path | None = None,
) -> pd.DataFrame:
    """Читает UNSMRY/SMSPEC прогона и возвращает таблицу в формате `field_rates`.

    `data_path` - путь к .DATA файлу кейса (симулятор пишет .SMSPEC/.UNSMRY
    рядом с ним, с тем же именем без расширения) - либо передайте
    `case_root` напрямую, если раскладка другая. `key_map` переопределяет
    набор читаемых summary-ключей (по умолчанию :data:`DEFAULT_KEY_MAP`) -
    берутся только те ключи из него, что реально есть в файле.
    """
    try:
        from resdata.summary import Summary
    except ImportError as exc:  # pragma: no cover - зависит от окружения пользователя
        raise ImportError(
            "Для чтения UNSMRY нужен пакет 'resdata' (pip install resdata) - "
            "он не входит в основной requirements.txt, так как нужен только для этого шага."
        ) from exc

    key_map = key_map if key_map is not None else DEFAULT_KEY_MAP
    root = str(case_root) if case_root is not None else str(Path(data_path).with_suffix(""))

    summary = Summary(root)
    available = [key for key in key_map if summary.has_key(key)]
    if not available:
        raise ValueError(f"Ни один из ключей {list(key_map)} не найден в {root}.UNSMRY")

    frame = summary.pandas_frame(column_keys=available)
    frame = frame.rename(columns={key: key_map[key] for key in available})
    frame.index.name = "date"
    return frame.reset_index()
