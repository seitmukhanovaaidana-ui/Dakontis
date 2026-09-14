"""Инструменты для работы с SCHEDULE-секцией Eclipse/Petrel (*.INC, *.DATA).

Основной вход - :func:`eclipse_schedule.parser.parse_schedule`, который
превращает `FINAL_SCH.INC`-подобный файл в таблицы pandas (скважины,
перфорации, история дебитов/закачки) без запуска самого симулятора.
"""

from .parser import ScheduleData, parse_schedule

__all__ = ["ScheduleData", "parse_schedule"]
