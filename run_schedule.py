#!/usr/bin/env python3
"""
CLI для разбора SCHEDULE-секции Eclipse/Petrel (например `FINAL_SCH.INC`)
в таблицы pandas, QC подозрительных дебитов, восстановления дебитов поля
и (опционально) обрезки истории для blind-test проверки адаптации.

Пример запуска:
    python run_schedule.py --input FINAL_SCH.INC --start-date 1973-06-01 --output-dir results_sch

Обрезать историю по дате (для blind-test):
    python run_schedule.py --input FINAL_SCH.INC --start-date 1973-06-01 \
        --split-date 2015-01-01 --split-output FINAL_SCH_history.INC
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eclipse_schedule.diagnostics import field_rates, voidage_replacement_ratio
from eclipse_schedule.parser import parse_schedule
from eclipse_schedule.qc import detect_placeholder_rates
from eclipse_schedule.report import plot_field_rates, save_tables
from eclipse_schedule.splitter import truncate_schedule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Путь к файлу SCHEDULE (например FINAL_SCH.INC)")
    p.add_argument("--output-dir", default="results_sch", help="Папка для результатов (по умолч. results_sch)")
    p.add_argument("--start-date", default=None,
                   help="Дата START из основного .DATA файла, например 1973-06-01. "
                        "Нужна, чтобы верно датировать самые первые скважины/дебиты "
                        "(до первого DATES).")
    p.add_argument("--min-wells", type=int, default=2,
                   help="QC: минимум скважин с одинаковым дебитом в один месяц, "
                        "чтобы считать это подозрительным (по умолч. 2)")
    p.add_argument("--vrr-window", type=int, default=12,
                   help="Окно скользящего среднего VRR в месяцах (по умолч. 12, 0 - отключить)")
    p.add_argument("--split-date", default=None,
                   help="Если задано - дополнительно сохранить обрезанную по этой дате "
                        "копию файла (blind-test: адаптация только до этой даты)")
    p.add_argument("--split-output", default=None,
                   help="Путь для обрезанного файла (по умолч. <output-dir>/history_until_<split-date>.INC)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    print(f"Читаю SCHEDULE: {args.input}")
    data = parse_schedule(args.input, start_date=args.start_date)
    print(f"  скважин (записей WELSPECS): {len(data.wells)}")
    print(f"  перфораций (записей COMPDAT): {len(data.completions)}")
    print(f"  строк истории (WCONHIST+WCONINJH): {len(data.history)}")
    print(f"  дат в DATES: {len(data.dates)}")

    print("Ищу подозрительные (одинаковые у нескольких скважин) дебиты ...")
    qc = detect_placeholder_rates(data.history, min_wells=args.min_wells)
    if qc.empty:
        print("  подозрительных дебитов не найдено")
    else:
        print(f"  найдено подозрительных групп: {len(qc)}")
        print(qc.head(10).to_string(index=False))

    print("Восстанавливаю дебиты по полю из истории ...")
    field = field_rates(data.history)
    window = args.vrr_window or None
    field = voidage_replacement_ratio(field, window=window)

    paths = save_tables(data, field, qc, out_dir)
    for name, path in paths.items():
        print(f"  {name}: {path}")

    plot_path = plot_field_rates(field, out_dir / "field_rates_plot.png")
    print(f"График сохранён: {plot_path}")

    if args.split_date:
        split_output = Path(args.split_output) if args.split_output else out_dir / f"history_until_{args.split_date}.INC"
        last_date = truncate_schedule(args.input, split_output, args.split_date)
        print(f"\nОбрезанный файл (blind-test) сохранён: {split_output}")
        print(f"  последняя оставленная дата: {last_date}")


if __name__ == "__main__":
    main()
