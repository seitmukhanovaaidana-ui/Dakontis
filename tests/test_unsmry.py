"""Тест read_field_rates() на настоящем (не поддельном) бинарном UNSMRY.

`resdata` умеет не только читать, но и писать summary-файлы
(`Summary.from_pandas` + `fwrite`) - это используется здесь, чтобы
сгенерировать реальный UNSMRY/SMSPEC и прогнать его через
`read_field_rates()`, не имея под рукой лицензии на симулятор. Если
`resdata` не установлен (это необязательная зависимость), тест
пропускается.
"""

from __future__ import annotations

import pandas as pd
import pytest

resdata = pytest.importorskip("resdata")

from history_match.unsmry import read_field_rates  # noqa: E402


def _write_synthetic_unsmry(tmp_path, case_name: str, frame: pd.DataFrame):
    import os

    from resdata.summary import Summary

    summary = Summary.from_pandas(case_name, frame)
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        summary.fwrite()
    finally:
        os.chdir(original_cwd)


def test_read_field_rates_roundtrip(tmp_path):
    dates = pd.date_range("2020-01-01", periods=3, freq="MS")
    frame = pd.DataFrame(
        {"FOPR": [100.0, 110.0, 120.0], "FWPR": [10.0, 12.0, 15.0], "FLPR": [110.0, 122.0, 135.0]},
        index=dates,
    )
    _write_synthetic_unsmry(tmp_path, "CASE000", frame)

    data_path = tmp_path / "CASE000.DATA"
    data_path.write_text("RUNSPEC\n", encoding="utf-8")  # сам симулятор не запускается, файл нужен только для пути

    result = read_field_rates(data_path)

    assert set(result.columns) == {"date", "liquid_rate", "oil_rate", "water_rate"}
    result = result.sort_values("date").reset_index(drop=True)
    assert result["date"].tolist() == list(dates)
    assert result["oil_rate"].tolist() == pytest.approx([100.0, 110.0, 120.0])
    assert result["water_rate"].tolist() == pytest.approx([10.0, 12.0, 15.0])
    assert result["liquid_rate"].tolist() == pytest.approx([110.0, 122.0, 135.0])


def test_read_field_rates_missing_keys_raises(tmp_path):
    dates = pd.date_range("2020-01-01", periods=2, freq="MS")
    frame = pd.DataFrame({"FGOR": [1.0, 2.0]}, index=dates)
    _write_synthetic_unsmry(tmp_path, "CASE001", frame)

    data_path = tmp_path / "CASE001.DATA"
    data_path.write_text("RUNSPEC\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_field_rates(data_path)


def test_read_field_rates_custom_key_map(tmp_path):
    dates = pd.date_range("2020-01-01", periods=2, freq="MS")
    frame = pd.DataFrame({"FWIR": [50.0, 60.0]}, index=dates)
    _write_synthetic_unsmry(tmp_path, "CASE002", frame)

    data_path = tmp_path / "CASE002.DATA"
    data_path.write_text("RUNSPEC\n", encoding="utf-8")

    result = read_field_rates(data_path, key_map={"FWIR": "injection_rate"})
    assert result["injection_rate"].tolist() == pytest.approx([50.0, 60.0])
