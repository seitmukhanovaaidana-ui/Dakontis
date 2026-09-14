#!/usr/bin/env python3
"""
CLI для ассистированной адаптации (history matching) через ES-MDA поверх
реального симулятора. Работает в два шага, потому что сам симулятор
(Eclipse/tNavigator/OPM Flow) здесь не запускается - его нужно прогнать
на кейсах отдельно, там, где он установлен и лицензирован:

1) init - строит начальный ансамбль кейсов (мультипликаторы по регионам
   из FINAL_PROP_EQLNUM.GRDECL/FIPNUM/...) рядом с основным .DATA файлом:

   python run_history_match.py init \\
       --data FINAL.DATA --region-file FINAL_PROP_EQLNUM.GRDECL \\
       --properties PERMX,PERMZ,PORO --n-realizations 50 \\
       --low 0.3 --high 3.0 --run-dir hm_run

   Прогоните реальный симулятор на каждом кейсе из hm_run/iteration_0/manifest.csv,
   приведите результат к таблице (date, liquid_rate, water_rate, ...) -
   как в eclipse_schedule.diagnostics.field_rates() - например, прочитав
   UNSMRY через resdata/ecl2df, и сохраните как
   hm_run/iteration_0/predictions/case_000.csv, case_001.csv, ...

2) update - один шаг ES-MDA по вашим результатам, готовит следующий
   ансамбль кейсов:

   python run_history_match.py update \\
       --run-dir hm_run --iteration 0 --n-iterations 4 \\
       --data FINAL.DATA --observed field_rates.csv \\
       --predictions-dir hm_run/iteration_0/predictions

   Повторяйте update для iteration 1, 2, ... (n_iterations-1), каждый раз
   прогоняя симулятор на новом hm_run/iteration_N/manifest.csv, пока не
   дойдёте до последней запланированной итерации.

Если ваш симулятор пишет обычный UNSMRY (Eclipse/tNavigator/OPM Flow),
шаг "приведите результат к таблице" из init можно не делать руками -
после прогона всех кейсов итерации вызовите:

   python run_history_match.py collect --run-dir hm_run --iteration 0

это прочитает .UNSMRY каждого кейса (через пакет `resdata`,
`pip install resdata`) и само разложит их по
hm_run/iteration_0/predictions/case_NNN.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from history_match.ensemble import build_ensemble
from history_match.es_mda import alpha_schedule, es_mda_update
from history_match.objective import ensemble_misfits, observation_vector, predictions_matrix
from history_match.parameters import ParameterSpace
from history_match.regions import parse_region_ids, read_dims


def _space_path(run_dir: Path) -> Path:
    return run_dir / "parameter_space.json"


def _iteration_dir(run_dir: Path, iteration: int) -> Path:
    return run_dir / f"iteration_{iteration}"


def _save_parameter_space(run_dir: Path, space: ParameterSpace) -> None:
    _space_path(run_dir).write_text(
        json.dumps({"region_ids": list(space.region_ids), "properties": list(space.properties)}, indent=2),
        encoding="utf-8",
    )


def _load_parameter_space(run_dir: Path) -> ParameterSpace:
    data = json.loads(_space_path(run_dir).read_text(encoding="utf-8"))
    return ParameterSpace(region_ids=tuple(data["region_ids"]), properties=tuple(data["properties"]))


def _save_manifest(iteration_dir: Path, paths: list[Path]) -> None:
    pd.DataFrame({"case_id": range(len(paths)), "data_path": [str(p) for p in paths]}).to_csv(
        iteration_dir / "manifest.csv", index=False
    )


def _load_manifest(iteration_dir: Path) -> pd.DataFrame:
    return pd.read_csv(iteration_dir / "manifest.csv")


def cmd_init(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    dims = read_dims(data_path)
    region_ids = sorted(parse_region_ids(args.region_file, args.region_keyword))
    properties = tuple(p.strip() for p in args.properties.split(","))

    space = ParameterSpace(region_ids=tuple(region_ids), properties=properties)
    log_vectors = space.sample_prior_log(args.n_realizations, args.low, args.high, seed=args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_parameter_space(run_dir, space)

    iteration_dir = _iteration_dir(run_dir, 0)
    iteration_dir.mkdir(parents=True, exist_ok=True)
    np.save(iteration_dir / "log_vectors.npy", log_vectors)

    case_dir = Path(args.case_dir) if args.case_dir else None
    paths = build_ensemble(
        data_path, dims, space, log_vectors,
        out_dir=case_dir, region_keyword=args.region_keyword, section=args.section, tag="it0",
    )
    _save_manifest(iteration_dir, paths)
    (iteration_dir / "predictions").mkdir(exist_ok=True)

    print(f"Регионов: {len(region_ids)} | свойств: {properties} | параметров ансамбля: {space.n_params}")
    print(f"Построено кейсов: {len(paths)} в {paths[0].parent}")
    print(f"Список кейсов: {iteration_dir / 'manifest.csv'}")
    print(
        "\nДальше: прогоните ваш симулятор на каждом .DATA из manifest.csv, приведите "
        "результат к таблице (date, liquid_rate, water_rate, ...) и сохраните как "
        f"{iteration_dir / 'predictions'}/case_NNN.csv (NNN - 3-значный case_id), "
        "затем запустите:\n"
        f"  python run_history_match.py update --run-dir {run_dir} --iteration 0 "
        f"--data {data_path} --observed <ваш факт>.csv --predictions-dir {iteration_dir / 'predictions'}"
    )


def cmd_collect(args: argparse.Namespace) -> None:
    from history_match.unsmry import DEFAULT_KEY_MAP, read_field_rates

    iteration_dir = _iteration_dir(Path(args.run_dir), args.iteration)
    manifest = _load_manifest(iteration_dir)
    predictions_dir = iteration_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    key_map = DEFAULT_KEY_MAP
    if args.keys:
        requested = [k.strip() for k in args.keys.split(",")]
        key_map = {k: DEFAULT_KEY_MAP.get(k, k.lower()) for k in requested}

    ok, failed = 0, []
    for case_id, data_path in zip(manifest["case_id"], manifest["data_path"]):
        try:
            frame = read_field_rates(data_path, key_map=key_map)
        except Exception as exc:  # noqa: BLE001 - показываем, какой именно кейс не прочитался, и продолжаем
            failed.append((case_id, str(exc)))
            continue
        frame.to_csv(predictions_dir / f"case_{case_id:03d}.csv", index=False)
        ok += 1

    print(f"Прочитано кейсов: {ok}/{len(manifest)} -> {predictions_dir}")
    if failed:
        print("Не удалось прочитать:")
        for case_id, error in failed:
            print(f"  case_{case_id:03d}: {error}")


def cmd_update(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    space = _load_parameter_space(run_dir)
    iteration_dir = _iteration_dir(run_dir, args.iteration)
    log_vectors = np.load(iteration_dir / "log_vectors.npy")
    manifest = _load_manifest(iteration_dir)

    observed = pd.read_csv(args.observed, parse_dates=["date"])
    value_cols = tuple(c.strip() for c in args.value_cols.split(","))

    predictions_dir = Path(args.predictions_dir)
    simulated_by_case: dict[int, pd.DataFrame] = {}
    for case_id in manifest["case_id"]:
        path = predictions_dir / f"case_{case_id:03d}.csv"
        if not path.exists():
            raise SystemExit(f"Не найден результат для кейса {case_id}: {path}")
        simulated_by_case[case_id] = pd.read_csv(path, parse_dates=["date"])

    misfits = ensemble_misfits(observed, simulated_by_case, value_cols)
    print("Невязка по кейсам (меньше - лучше), топ-5 лучших:")
    print(misfits.head(5).to_string())

    dates = sorted(observed["date"].unique())
    obs_vector, obs_std = observation_vector(observed, dates, value_cols, obs_std_frac=args.obs_std_frac)
    ordered_predictions = [simulated_by_case[i] for i in manifest["case_id"]]
    pred_matrix = predictions_matrix(ordered_predictions, dates, value_cols)

    alpha = alpha_schedule(args.n_iterations)[args.iteration]
    rng = np.random.default_rng(args.seed)
    updated = es_mda_update(log_vectors, pred_matrix, obs_vector, obs_std, alpha, rng=rng)

    next_iteration = args.iteration + 1
    next_dir = _iteration_dir(run_dir, next_iteration)
    next_dir.mkdir(parents=True, exist_ok=True)
    np.save(next_dir / "log_vectors.npy", updated)

    data_path = Path(args.data)
    dims = read_dims(data_path)
    case_dir = Path(args.case_dir) if args.case_dir else None
    paths = build_ensemble(
        data_path, dims, space, updated,
        out_dir=case_dir, region_keyword=args.region_keyword, section=args.section, tag=f"it{next_iteration}",
    )
    _save_manifest(next_dir, paths)
    (next_dir / "predictions").mkdir(exist_ok=True)

    print(f"\nИтерация {next_iteration}: построено кейсов {len(paths)} в {paths[0].parent}")
    if next_iteration >= args.n_iterations:
        print("Это была последняя запланированная итерация ES-MDA.")
    else:
        print(
            f"Прогоните симулятор на {next_dir / 'manifest.csv'}, сохраните результаты в "
            f"{next_dir / 'predictions'}/case_NNN.csv, затем запустите update --iteration {next_iteration}."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Построить начальный ансамбль кейсов")
    init_p.add_argument("--data", required=True, help="Основной .DATA файл модели")
    init_p.add_argument("--region-file", required=True, help="GRDECL файл региона (например FINAL_PROP_EQLNUM.GRDECL)")
    init_p.add_argument("--region-keyword", default="EQLNUM")
    init_p.add_argument("--properties", default="PERMX,PERMY,PERMZ,PORO", help="Через запятую")
    init_p.add_argument("--n-realizations", type=int, default=50)
    init_p.add_argument("--low", type=float, default=0.3, help="Нижняя граница мультипликатора")
    init_p.add_argument("--high", type=float, default=3.0, help="Верхняя граница мультипликатора")
    init_p.add_argument("--seed", type=int, default=None)
    init_p.add_argument("--run-dir", required=True, help="Папка для метаданных ансамбля (не для .DATA-кейсов)")
    init_p.add_argument("--case-dir", default=None, help="Куда класть .DATA-кейсы (по умолч. рядом с --data)")
    init_p.add_argument("--section", default="EDIT", help="В какую секцию вставлять INCLUDE с мультипликаторами")
    init_p.set_defaults(func=cmd_init)

    collect_p = sub.add_parser(
        "collect", help="Прочитать UNSMRY кейсов и разложить по predictions/case_NNN.csv (нужен pip install resdata)"
    )
    collect_p.add_argument("--run-dir", required=True)
    collect_p.add_argument("--iteration", type=int, required=True)
    collect_p.add_argument(
        "--keys", default=None,
        help="Через запятую, какие summary-ключи читать (по умолч. FLPR,FOPR,FWPR,FGPR,FWIR - см. unsmry.DEFAULT_KEY_MAP)",
    )
    collect_p.set_defaults(func=cmd_collect)

    update_p = sub.add_parser("update", help="Один шаг ES-MDA по результатам прогонов симулятора")
    update_p.add_argument("--run-dir", required=True)
    update_p.add_argument("--iteration", type=int, required=True, help="Номер текущей итерации (с 0)")
    update_p.add_argument("--n-iterations", type=int, default=4, help="Всего запланировано итераций ES-MDA")
    update_p.add_argument("--observed", required=True, help="CSV с фактом: date, liquid_rate, water_rate, ...")
    update_p.add_argument("--predictions-dir", required=True, help="Папка с case_NNN.csv по каждому кейсу")
    update_p.add_argument("--value-cols", default="liquid_rate,water_rate")
    update_p.add_argument("--obs-std-frac", type=float, default=0.1)
    update_p.add_argument("--data", required=True, help="Основной .DATA файл модели (для кейсов следующей итерации)")
    update_p.add_argument("--region-keyword", default="EQLNUM")
    update_p.add_argument("--case-dir", default=None)
    update_p.add_argument("--section", default="EDIT")
    update_p.add_argument("--seed", type=int, default=None)
    update_p.set_defaults(func=cmd_update)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
