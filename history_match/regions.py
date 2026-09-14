"""Чтение регионов (EQLNUM/FIPNUM/...) и генерация мультипликаторов по ним.

Файлы вида `FINAL_PROP_EQLNUM.GRDECL` содержат один массив на ячейку
сетки (обычно миллионы значений), закодированный с повторами Eclipse
(`19*1` значит "19 раз подряд 1"). Чтобы понять, какие регионы вообще
есть в модели, не нужно разворачивать весь массив - достаточно
собрать множество встретившихся значений.

Применить мультипликатор к свойству только в ячейках одного региона
можно без переписывания самого массива PERMX/PORO - ключевые слова
EQUALS/ADD/MULTIPLY/COPY в Eclipse поддерживают необязательные
последние два параметра записи: номер региона и имя регион-массива,
которые ограничивают операцию (заданную боксом I/J/K) только теми
ячейками бокса, где этот регион-массив равен указанному номеру. Это и
используется здесь: бокс берётся равным всей сетке, а фильтрация - по
региону, так что реальные (потенциально несвязные) границы региона не
нужно вычислять самому.
"""

from __future__ import annotations

import re
from pathlib import Path

_TOKEN_RE = re.compile(r"(\d+)\*(-?\d+)|(-?\d+)")


def _strip_comment(line: str) -> str:
    idx = line.find("--")
    return line if idx == -1 else line[:idx]


def parse_region_ids(path: str | Path, keyword: str) -> set[int]:
    """Возвращает множество различных значений региона в GRDECL-файле.

    `keyword` - имя ключевого слова массива (например 'EQLNUM', 'FIPNUM').
    Не разворачивает массив целиком - только собирает уникальные значения
    из записей вида `N*value` или `value`, что дёшево даже для сеток на
    десятки миллионов ячеек.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    in_block = False
    values: set[int] = set()
    for raw_line in lines:
        line = _strip_comment(raw_line).strip()
        if not in_block:
            if line == keyword:
                in_block = True
            continue

        if not line:
            continue
        ended = line.endswith("/")
        line = line[:-1] if ended else line
        for match in _TOKEN_RE.finditer(line):
            repeat, repeat_value, plain_value = match.groups()
            values.add(int(repeat_value if repeat_value is not None else plain_value))
        if ended:
            break

    if not values:
        raise ValueError(f"Ключевое слово {keyword!r} не найдено (или пусто) в {path}")
    return values


def read_dims(data_path: str | Path) -> tuple[int, int, int]:
    """Читает NX, NY, NZ из ключевого слова DIMENS основного .DATA файла."""
    text = Path(data_path).read_text(encoding="utf-8", errors="replace")

    in_block = False
    numbers: list[int] = []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).strip()
        if not in_block:
            if line == "DIMENS":
                in_block = True
            continue
        if not line:
            continue
        ended = line.endswith("/")
        content = line[:-1] if ended else line
        numbers.extend(int(tok) for tok in content.split())
        if ended:
            break

    if len(numbers) != 3:
        raise ValueError(f"Не удалось прочитать DIMENS (NX NY NZ) из {data_path}")
    return numbers[0], numbers[1], numbers[2]


def generate_multiplier_include(
    multipliers: dict[int, dict[str, float]],
    dims: tuple[int, int, int],
    out_path: str | Path,
    region_keyword: str = "EQLNUM",
) -> Path:
    """Пишет INCLUDE-файл с MULTIPLY-записями: по одной на (регион, свойство).

    `dims` - (NX, NY, NZ) из DIMENS основного .DATA файла - бокс операции
    берётся равным всей сетке, а фильтрация по региону задаётся последними
    двумя параметрами записи MULTIPLY (номер региона, имя регион-массива).
    Файл нужно подключить (INCLUDE) в секции EDIT или GRID основного
    .DATA файла - после того, как соответствующие свойства и регион-массив
    уже загружены (см. :func:`history_match.ensemble.build_case`).
    """
    nx, ny, nz = dims
    out_path = Path(out_path)

    lines = ["MULTIPLY"]
    for region in sorted(multipliers):
        for prop, factor in multipliers[region].items():
            lines.append(
                f"  '{prop}' {factor:.6f}  1 {nx}  1 {ny}  1 {nz}  {region} '{region_keyword}' /"
            )
    lines.append("/")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
