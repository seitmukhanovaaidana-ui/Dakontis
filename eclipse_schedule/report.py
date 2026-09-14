"""Сохранение таблиц (CSV) и графика дебитов поля на диск."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .parser import ScheduleData


def save_tables(data: ScheduleData, field: pd.DataFrame, qc: pd.DataFrame, out_dir: str | Path) -> dict[str, Path]:
    """Сохраняет wells/completions/history/field_rates/qc в CSV.

    Возвращает словарь {имя_таблицы: путь}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "wells": out_dir / "wells.csv",
        "completions": out_dir / "completions.csv",
        "history": out_dir / "history.csv",
        "field_rates": out_dir / "field_rates.csv",
        "qc_suspects": out_dir / "qc_suspect_rates.csv",
    }

    data.wells.to_csv(paths["wells"], index=False)
    data.completions.to_csv(paths["completions"], index=False)
    data.history.to_csv(paths["history"], index=False)
    field.to_csv(paths["field_rates"], index=False)
    qc.to_csv(paths["qc_suspects"], index=False)

    return paths


def plot_field_rates(field: pd.DataFrame, out_path: str | Path) -> Path:
    """Строит график дебитов поля (жидкость/нефть/вода) по датам."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(field["date"], field["liquid_rate"], color="magenta", linewidth=1, label="Liquid rate (из SCH.INC)")
    ax.plot(field["date"], field["oil_rate"], color="green", linewidth=1, label="Oil rate")
    if "injection_rate" in field.columns and field["injection_rate"].abs().sum() > 0:
        ax.plot(field["date"], field["injection_rate"], color="blue", linewidth=1, label="Injection rate")

    ax.set_xlabel("Date")
    ax.set_ylabel("Rate, sm3/d")
    ax.set_title("Field rates, восстановлено из FINAL_SCH.INC")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
