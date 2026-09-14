"""
Настольное приложение (графический интерфейс) для eclipse_schedule.

Загружаете файл SCHEDULE (например `FINAL_SCH.INC`), указываете дату
START - программа разбирает историю, ищет подозрительные ("заглушечные")
дебиты, строит график дебитов по полю и позволяет сохранить обрезанную
по дате копию файла для blind-test проверки будущей адаптации.

Запуск: python run_gui.py
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .diagnostics import field_rates, voidage_replacement_ratio
from .parser import ScheduleData, parse_schedule
from .qc import detect_placeholder_rates
from .report import save_tables
from .splitter import truncate_schedule

ALL = "Все"
WELL_COLUMNS = ("well", "group", "i", "j", "phase", "date")
QC_COLUMNS = ("date", "field", "value", "n_wells", "wells")


class ScheduleApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Разбор SCHEDULE и QC истории (eclipse_schedule)")
        self.root.geometry("1300x820")

        self.path: Path | None = None
        self.data: ScheduleData | None = None
        self.field: pd.DataFrame | None = None
        self.qc: pd.DataFrame | None = None

        self._build_widgets()

    # ------------------------------------------------------------------ UI

    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="Загрузить SCHEDULE...", command=self.on_load).pack(side="left")
        ttk.Label(top, text="Дата START:").pack(side="left", padx=(14, 2))
        self.start_date_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.start_date_var, width=12).pack(side="left")
        ttk.Label(top, text="(например 1973-06-01)", foreground="gray").pack(side="left", padx=(4, 0))

        self.file_label = ttk.Label(top, text="Файл не загружен")
        self.file_label.pack(side="left", padx=14)

        ttk.Button(top, text="Экспортировать результаты...", command=self.on_export).pack(side="right")

        self.summary_label = tk.Label(
            self.root,
            text="Загрузите файл SCHEDULE (например FINAL_SCH.INC).",
            font=("Segoe UI", 12, "bold"),
            bg="#ED7D31",
            fg="white",
            anchor="w",
            padx=12,
            pady=8,
        )
        self.summary_label.pack(fill="x", padx=8, pady=6)

        self._build_split_panel(self.root)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=4)
        self.notebook = notebook

        chart_tab = ttk.Frame(notebook)
        qc_tab = ttk.Frame(notebook)
        wells_tab = ttk.Frame(notebook)
        well_tab = ttk.Frame(notebook)
        notebook.add(chart_tab, text="Дебиты по полю")
        notebook.add(qc_tab, text="QC (подозрительные дебиты)")
        notebook.add(wells_tab, text="Скважины")
        notebook.add(well_tab, text="История по скважине")

        self._build_chart_tab(chart_tab)
        self._build_qc_tab(qc_tab)
        self._build_wells_tab(wells_tab)
        self._build_well_history_tab(well_tab)

    def _build_split_panel(self, parent: tk.Widget) -> None:
        panel = ttk.LabelFrame(parent, text="Blind-test: обрезать историю по дате", padding=8)
        panel.pack(fill="x", padx=8, pady=(0, 4))

        ttk.Label(panel, text="Обрезать после (YYYY-MM-DD):").pack(side="left")
        self.split_date_var = tk.StringVar(value="")
        ttk.Entry(panel, textvariable=self.split_date_var, width=12).pack(side="left", padx=6)
        ttk.Button(panel, text="Сохранить обрезанный файл...", command=self.on_split).pack(side="left")

    def _build_chart_tab(self, parent: ttk.Widget) -> None:
        controls = ttk.Frame(parent, padding=6)
        controls.pack(fill="x")
        ttk.Label(controls, text="Окно скольз. среднего VRR, мес:").pack(side="left")
        self.vrr_window_var = tk.StringVar(value="12")
        ttk.Entry(controls, textvariable=self.vrr_window_var, width=6).pack(side="left", padx=6)
        ttk.Button(controls, text="Обновить график", command=self.recompute_field).pack(side="left")
        ttk.Button(controls, text="Сохранить график...", command=self.on_save_chart).pack(side="right")

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_qc_tab(self, parent: ttk.Widget) -> None:
        controls = ttk.Frame(parent, padding=6)
        controls.pack(fill="x")
        ttk.Label(controls, text="Мин. скважин с одинак. дебитом:").pack(side="left")
        self.min_wells_var = tk.StringVar(value="2")
        ttk.Entry(controls, textvariable=self.min_wells_var, width=5).pack(side="left", padx=6)
        ttk.Label(controls, text="Показать поле:").pack(side="left", padx=(14, 2))
        self.qc_field_var = tk.StringVar(value=ALL)
        self.qc_field_combo = ttk.Combobox(
            controls,
            textvariable=self.qc_field_var,
            state="readonly",
            values=[ALL, "oil_rate", "water_rate", "gas_rate"],
            width=12,
        )
        self.qc_field_combo.pack(side="left")
        self.qc_field_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_qc_table())
        ttk.Button(controls, text="Обновить QC", command=self.recompute_qc).pack(side="left", padx=(14, 0))

        self.qc_count_label = ttk.Label(parent, text="")
        self.qc_count_label.pack(fill="x", padx=6)

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, padx=(6, 0), pady=(0, 6))
        self.qc_tree = ttk.Treeview(body, columns=QC_COLUMNS, show="headings")
        widths = {"date": 90, "field": 90, "value": 90, "n_wells": 80, "wells": 500}
        for col in QC_COLUMNS:
            self.qc_tree.heading(col, text=col)
            self.qc_tree.column(col, width=widths.get(col, 100), anchor="w" if col == "wells" else "center")
        vsb = ttk.Scrollbar(body, orient="vertical", command=self.qc_tree.yview)
        self.qc_tree.configure(yscrollcommand=vsb.set)
        self.qc_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")

    def _build_wells_tab(self, parent: ttk.Widget) -> None:
        self.wells_tree = ttk.Treeview(parent, columns=WELL_COLUMNS, show="headings")
        for col in WELL_COLUMNS:
            self.wells_tree.heading(col, text=col)
            self.wells_tree.column(col, width=110, anchor="center")
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.wells_tree.yview)
        self.wells_tree.configure(yscrollcommand=vsb.set)
        self.wells_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")

    def _build_well_history_tab(self, parent: ttk.Widget) -> None:
        controls = ttk.Frame(parent, padding=6)
        controls.pack(fill="x")
        ttk.Label(controls, text="Скважина:").pack(side="left")
        self.well_pick_var = tk.StringVar(value="")
        self.well_pick_combo = ttk.Combobox(controls, textvariable=self.well_pick_var, state="readonly", width=15)
        self.well_pick_combo.pack(side="left", padx=6)
        self.well_pick_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_well_history_plot())

        self.well_figure = Figure(figsize=(8, 4.5), dpi=100)
        self.well_ax = self.well_figure.add_subplot(111)
        self.well_canvas = FigureCanvasTkAgg(self.well_figure, master=parent)
        self.well_canvas.get_tk_widget().pack(fill="both", expand=True)

    # --------------------------------------------------------------- load

    def on_load(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл SCHEDULE",
            filetypes=[("Eclipse INC/DATA", "*.INC *.inc *.DATA *.data"), ("Все файлы", "*.*")],
        )
        if not path:
            return

        start_date = self.start_date_var.get().strip() or None
        try:
            data = parse_schedule(path, start_date=start_date)
        except Exception as exc:  # noqa: BLE001 - показываем пользователю любую ошибку разбора
            messagebox.showerror("Ошибка разбора", str(exc))
            return

        if data.history.empty:
            messagebox.showwarning(
                "Пусто",
                "В файле не найдено ни одной записи WCONHIST/WCONINJH - проверьте, что это файл SCHEDULE.",
            )
            return

        self.path = Path(path)
        self.data = data
        n_wells = data.wells["well"].nunique() if not data.wells.empty else 0
        self.file_label.config(text=f"{self.path.name}  (скважин: {n_wells}, дат: {len(data.dates)})")

        self._update_wells_table()
        wells = sorted(data.history["well"].dropna().astype(str).unique())
        self.well_pick_combo["values"] = wells
        if wells:
            self.well_pick_var.set(wells[0])

        self.recompute_qc()
        self.recompute_field()
        self._update_well_history_plot()

    # ---------------------------------------------------------- recompute

    def recompute_qc(self) -> None:
        if self.data is None:
            return
        try:
            min_wells = int(self.min_wells_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "'Мин. скважин' должно быть целым числом.")
            return
        self.qc = detect_placeholder_rates(self.data.history, min_wells=min_wells)
        self._refresh_qc_table()
        self._update_summary()

    def _refresh_qc_table(self) -> None:
        self.qc_tree.delete(*self.qc_tree.get_children())
        if self.qc is None:
            self.qc_count_label.config(text="")
            return
        qc = self.qc
        field = self.qc_field_var.get()
        if field != ALL:
            qc = qc[qc["field"] == field]
        self.qc_count_label.config(text=f"Найдено подозрительных групп: {len(qc)}")
        for _, row in qc.head(1000).iterrows():
            wells_str = ", ".join(str(w) for w in row["wells"])
            date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else ""
            self.qc_tree.insert(
                "", "end", values=(date_str, row["field"], f"{row['value']:.6g}", row["n_wells"], wells_str)
            )

    def recompute_field(self) -> None:
        if self.data is None:
            return
        try:
            window = int(self.vrr_window_var.get()) or None
        except ValueError:
            window = None
        try:
            field = field_rates(self.data.history)
        except ValueError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return
        self.field = voidage_replacement_ratio(field, window=window)
        self._update_chart()
        self._update_summary()

    def _update_summary(self) -> None:
        if self.data is None or self.field is None or self.field.empty:
            return
        f = self.field
        n_wells = self.data.wells["well"].nunique() if not self.data.wells.empty else 0
        n_qc = 0 if self.qc is None else len(self.qc)
        self.summary_label.config(
            text=(
                f"Скважин: {n_wells} | Перфораций: {len(self.data.completions)} | "
                f"Дат: {len(self.data.dates)} ({f['date'].min():%Y-%m-%d} - {f['date'].max():%Y-%m-%d}) | "
                f"Подозрительных QC-групп: {n_qc}"
            )
        )

    def _update_chart(self) -> None:
        self.ax.clear()
        field = self.field
        self.ax.plot(field["date"], field["liquid_rate"], color="magenta", linewidth=1, label="Liquid rate")
        self.ax.plot(field["date"], field["oil_rate"], color="green", linewidth=1, label="Oil rate")
        if field["injection_rate"].abs().sum() > 0:
            self.ax.plot(field["date"], field["injection_rate"], color="blue", linewidth=1, label="Injection rate")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Rate, sm3/d")
        self.ax.set_title("Field rates, восстановлено из SCHEDULE")
        self.ax.legend(fontsize=9, loc="best")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def _update_wells_table(self) -> None:
        self.wells_tree.delete(*self.wells_tree.get_children())
        if self.data is None:
            return
        for _, row in self.data.wells.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else ""
            self.wells_tree.insert(
                "", "end", values=(row["well"], row["group"], row["i"], row["j"], row["phase"], date_str)
            )

    def _update_well_history_plot(self) -> None:
        self.well_ax.clear()
        well = self.well_pick_var.get()
        if self.data is not None and well:
            h = self.data.history[self.data.history["well"] == well]
            prod = h[h["kind"] == "producer"].sort_values("date")
            inj = h[h["kind"] == "injector"].sort_values("date")
            if not prod.empty:
                self.well_ax.plot(prod["date"], prod["oil_rate"], color="green", label="Oil rate")
                self.well_ax.plot(prod["date"], prod["water_rate"], color="blue", label="Water rate")
            if not inj.empty:
                self.well_ax.plot(inj["date"], inj["surface_rate"], color="red", label="Injection rate")
            self.well_ax.set_title(f"История скважины {well}")
        self.well_ax.set_xlabel("Date")
        self.well_ax.set_ylabel("Rate, sm3/d")
        self.well_ax.legend(fontsize=9, loc="best")
        self.well_ax.grid(True, alpha=0.3)
        self.well_canvas.draw()

    # ------------------------------------------------------------ actions

    def on_split(self) -> None:
        if self.path is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите файл SCHEDULE.")
            return
        cutoff = self.split_date_var.get().strip()
        if not cutoff:
            messagebox.showwarning("Нет даты", "Укажите дату обрезки (YYYY-MM-DD).")
            return

        out_path = filedialog.asksaveasfilename(
            title="Сохранить обрезанный файл как...",
            defaultextension=".INC",
            filetypes=[("Eclipse INC", "*.INC"), ("Все файлы", "*.*")],
            initialfile=f"{self.path.stem}_history.INC",
        )
        if not out_path:
            return

        try:
            last_date = truncate_schedule(self.path, out_path, cutoff)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка обрезки", str(exc))
            return

        messagebox.showinfo(
            "Готово", f"Обрезанный файл сохранён:\n{out_path}\n\nПоследняя оставленная дата: {last_date}"
        )

    def on_save_chart(self) -> None:
        if self.field is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить график как...",
            defaultextension=".png",
            filetypes=[("Изображение PNG", "*.png"), ("PDF", "*.pdf"), ("Все файлы", "*.*")],
            initialfile="field_rates_plot.png",
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка сохранения графика", str(exc))
            return
        messagebox.showinfo("Готово", f"График сохранён:\n{path}")

    def on_export(self) -> None:
        if self.data is None or self.field is None or self.qc is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите файл SCHEDULE.")
            return
        out_dir = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
        if not out_dir:
            return
        try:
            paths = save_tables(self.data, self.field, self.qc, out_dir)
            plot_path = Path(out_dir) / "field_rates_plot.png"
            self.figure.savefig(plot_path, dpi=200, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка экспорта", str(exc))
            return
        message = "\n".join(str(p) for p in paths.values()) + f"\n{plot_path}"
        messagebox.showinfo("Готово", f"Сохранено:\n{message}")


def main() -> None:
    root = tk.Tk()
    ScheduleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
