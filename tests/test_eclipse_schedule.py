"""Тесts для eclipse_schedule: парсер, QC, диагностика, обрезка по дате.

Используют examples/schedule_example.INC - маленький синтетический файл,
который повторяет структуру реального Petrel-экспорта (WELSPECS перед
первым DATES, довскрытия/новые скважины после DATES, WCONHIST/WCONINJH,
"шумовые" ключевые слова TUNING/WLIST/GRUPTREE/FBHPDEF между блоками).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eclipse_schedule.diagnostics import field_rates, voidage_replacement_ratio
from eclipse_schedule.parser import parse_schedule
from eclipse_schedule.qc import detect_placeholder_rates
from eclipse_schedule.splitter import truncate_schedule

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "schedule_example.INC"
START_DATE = "1980-01-01"


def test_parse_schedule_basic():
    data = parse_schedule(EXAMPLE, start_date=START_DATE)

    assert sorted(data.wells["well"]) == ["A1", "A2", "A3"]
    assert set(data.completions["well"]) == {"A1", "A2", "A3"}

    a3 = data.wells.set_index("well").loc["A3"]
    assert a3["date"] == pd.Timestamp("1980-02-01")

    a1_compdat = data.completions.set_index("well").loc["A1"]
    assert a1_compdat["k1"] == 100
    assert a1_compdat["k2"] == 101
    assert a1_compdat["conn_factor"] == pytest.approx(5.0)

    assert data.dates == [pd.Timestamp("1980-02-01"), pd.Timestamp("1980-03-01"), pd.Timestamp("1980-04-01")]

    producers = data.history[data.history["kind"] == "producer"]
    injectors = data.history[data.history["kind"] == "injector"]
    assert len(producers) == 2 + 3 + 3 + 3
    assert len(injectors) == 2

    jan_a1 = producers[(producers["well"] == "A1") & (producers["date"] == pd.Timestamp("1980-01-01"))].iloc[0]
    assert jan_a1["oil_rate"] == pytest.approx(50.0)
    assert jan_a1["cmode"] == "LRAT"

    mar_inj = injectors[injectors["date"] == pd.Timestamp("1980-03-01")].iloc[0]
    assert mar_inj["well"] == "A2"
    assert mar_inj["inj_type"] == "WATER"
    assert mar_inj["surface_rate"] == pytest.approx(25.0)


def test_detect_placeholder_rates_flags_shared_tiny_value():
    data = parse_schedule(EXAMPLE, start_date=START_DATE)
    qc = detect_placeholder_rates(data.history, min_wells=2)

    assert len(qc) == 1
    row = qc.iloc[0]
    assert row["date"] == pd.Timestamp("1980-02-01")
    assert row["field"] == "oil_rate"
    assert row["value"] == pytest.approx(0.114943)
    assert row["n_wells"] == 2
    assert set(row["wells"]) == {"A2", "A3"}


def test_detect_placeholder_rates_empty_when_stricter_threshold():
    data = parse_schedule(EXAMPLE, start_date=START_DATE)
    qc = detect_placeholder_rates(data.history, min_wells=3)
    assert qc.empty


def test_field_rates_and_vrr():
    data = parse_schedule(EXAMPLE, start_date=START_DATE)
    field = field_rates(data.history)

    by_date = field.set_index("date")
    jan = by_date.loc[pd.Timestamp("1980-01-01")]
    assert jan["oil_rate"] == pytest.approx(50.0 + 0.114943)
    assert jan["active_producers"] == 2
    assert jan["injection_rate"] == pytest.approx(0.0)

    mar = by_date.loc[pd.Timestamp("1980-03-01")]
    assert mar["oil_rate"] == pytest.approx(98.0)
    assert mar["water_rate"] == pytest.approx(3.7)
    assert mar["liquid_rate"] == pytest.approx(101.7)
    assert mar["injection_rate"] == pytest.approx(25.0)
    assert mar["active_injectors"] == 1

    vrr = voidage_replacement_ratio(field, window=2)
    vrr_by_date = vrr.set_index("date")
    assert vrr_by_date.loc[pd.Timestamp("1980-03-01"), "vrr"] == pytest.approx(25.0 / 101.7)
    assert "vrr_rolling" in vrr.columns


def test_truncate_schedule_keeps_only_history_up_to_cutoff(tmp_path):
    out_path = tmp_path / "history_until_1980-02-15.INC"
    last_date = truncate_schedule(EXAMPLE, out_path, cutoff_date="1980-02-15")

    assert last_date == pd.Timestamp("1980-02-01")
    assert out_path.exists()

    truncated = parse_schedule(out_path, start_date=START_DATE)
    assert truncated.dates == [pd.Timestamp("1980-02-01")]
    assert sorted(truncated.wells["well"]) == ["A1", "A2", "A3"]
    assert (truncated.history["kind"] == "injector").sum() == 0

    producers = truncated.history[truncated.history["kind"] == "producer"]
    assert len(producers) == 2 + 3


def test_truncate_schedule_keeps_everything_when_cutoff_after_last_date(tmp_path):
    out_path = tmp_path / "full_copy.INC"
    last_date = truncate_schedule(EXAMPLE, out_path, cutoff_date="1999-01-01")

    assert last_date == pd.Timestamp("1980-04-01")
    full = parse_schedule(out_path, start_date=START_DATE)
    assert len(full.dates) == 3
