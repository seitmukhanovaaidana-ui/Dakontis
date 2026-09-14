"""Тесты для history_match: parameters, regions, ensemble, objective, es_mda.

Реальный симулятор здесь нигде не запускается - только текстовая генерация
кейсов (regions/ensemble) и чистая математика (parameters/objective/es_mda),
проверяемая на синтетических/игрушечных примерах.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from history_match.ensemble import build_case, build_ensemble, run_case
from history_match.es_mda import alpha_schedule, es_mda_update, run_history_matching_loop
from history_match.objective import ensemble_misfits, misfit, observation_vector, predictions_matrix
from history_match.parameters import ParameterSpace
from history_match.regions import generate_multiplier_include, parse_region_ids, read_dims

EQLNUM_TEXT = """\
-- Generated : Petrel
EQLNUM
-- Property name in Petrel : Region
  3*1 2*2 3*1
  2*2 4*1
/
"""


# ------------------------------------------------------------ parameters


def test_parameter_space_roundtrip():
    space = ParameterSpace(region_ids=(1, 2), properties=("PERMX", "PORO"))
    assert space.n_params == 4
    assert space.keys == [(1, "PERMX"), (1, "PORO"), (2, "PERMX"), (2, "PORO")]

    log_vector = np.log([1.5, 0.9, 0.6, 1.2])
    multipliers = space.to_multiplier_dict(log_vector)
    assert multipliers == {1: {"PERMX": pytest.approx(1.5), "PORO": pytest.approx(0.9)},
                            2: {"PERMX": pytest.approx(0.6), "PORO": pytest.approx(1.2)}}

    back = space.from_multiplier_dict(multipliers)
    np.testing.assert_allclose(back, log_vector)


def test_parameter_space_sample_prior_log_shape_and_bounds():
    space = ParameterSpace(region_ids=(1, 2, 3), properties=("PERMX",))
    ensemble = space.sample_prior_log(n_realizations=50, low=0.5, high=2.0, seed=0)
    assert ensemble.shape == (3, 50)
    multipliers = np.exp(ensemble)
    assert multipliers.min() >= 0.5
    assert multipliers.max() <= 2.0


def test_sample_prior_log_rejects_bad_bounds():
    space = ParameterSpace(region_ids=(1,), properties=("PERMX",))
    with pytest.raises(ValueError):
        space.sample_prior_log(10, low=2.0, high=0.5)
    with pytest.raises(ValueError):
        space.sample_prior_log(10, low=-1.0, high=2.0)


# ---------------------------------------------------------------- regions


def test_parse_region_ids(tmp_path):
    path = tmp_path / "EQLNUM.GRDECL"
    path.write_text(EQLNUM_TEXT, encoding="utf-8")
    assert parse_region_ids(path, "EQLNUM") == {1, 2}


def test_parse_region_ids_missing_keyword(tmp_path):
    path = tmp_path / "EQLNUM.GRDECL"
    path.write_text(EQLNUM_TEXT, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_region_ids(path, "FIPNUM")


def test_generate_multiplier_include(tmp_path):
    out_path = tmp_path / "MULT.INC"
    generate_multiplier_include(
        {1: {"PERMX": 1.5}, 2: {"PERMX": 0.75, "PORO": 1.1}},
        dims=(10, 20, 30),
        out_path=out_path,
        region_keyword="EQLNUM",
    )
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("MULTIPLY")
    assert text.rstrip().endswith("/")
    assert "'PERMX' 1.500000  1 10  1 20  1 30  1 'EQLNUM' /" in text
    assert "'PERMX' 0.750000  1 10  1 20  1 30  2 'EQLNUM' /" in text
    assert "'PORO' 1.100000  1 10  1 20  1 30  2 'EQLNUM' /" in text


def test_read_dims(tmp_path):
    path = tmp_path / "FINAL.DATA"
    path.write_text("RUNSPEC\nDIMENS\n 123 188 582 /\nGRID\n", encoding="utf-8")
    assert read_dims(path) == (123, 188, 582)


def test_read_dims_missing_raises(tmp_path):
    path = tmp_path / "FINAL.DATA"
    path.write_text("RUNSPEC\nGRID\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_dims(path)


# --------------------------------------------------------------- ensemble

BASE_DATA_TEXT = """\
RUNSPEC
DIMENS
 2 2 2 /
GRID
INCLUDE
 'GRID.INC' /
ECHO
EDIT
PROPS
INCLUDE
 'PROPS.INC' /
"""


def test_build_case_injects_include_after_section(tmp_path):
    base_path = tmp_path / "FINAL.DATA"
    base_path.write_text(BASE_DATA_TEXT, encoding="utf-8")

    data_path = build_case(
        base_path, case_id=3, multipliers={1: {"PERMX": 1.5}, 2: {"PERMX": 0.7}}, dims=(2, 2, 2)
    )

    assert data_path.name == "FINAL_case003.DATA"
    assert data_path.parent == base_path.parent

    lines = data_path.read_text(encoding="utf-8").splitlines()
    edit_idx = lines.index("EDIT")
    assert lines[edit_idx + 1].startswith("INCLUDE")
    assert "FINAL_case003_MULT.INC" in lines[edit_idx + 2]

    # PROPS section (и всё, что после EDIT) должна остаться на месте
    assert "PROPS" in lines

    include_path = base_path.parent / "FINAL_case003_MULT.INC"
    assert include_path.exists()
    assert "1.500000" in include_path.read_text(encoding="utf-8")


def test_build_case_missing_section_raises(tmp_path):
    base_path = tmp_path / "FINAL.DATA"
    base_path.write_text("RUNSPEC\nGRID\nPROPS\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_case(base_path, 0, {1: {"PERMX": 1.0}}, dims=(2, 2, 2))


def test_build_ensemble_creates_one_case_per_column(tmp_path):
    base_path = tmp_path / "FINAL.DATA"
    base_path.write_text(BASE_DATA_TEXT, encoding="utf-8")

    space = ParameterSpace(region_ids=(1, 2), properties=("PERMX",))
    log_vectors = space.sample_prior_log(n_realizations=5, low=0.5, high=2.0, seed=1)

    paths = build_ensemble(base_path, dims=(2, 2, 2), parameter_space=space, log_vectors=log_vectors)

    assert len(paths) == 5
    assert [p.name for p in paths] == [f"FINAL_case{i:03d}.DATA" for i in range(5)]
    for p in paths:
        assert p.exists()


def test_build_case_tag_avoids_collision_across_iterations(tmp_path):
    base_path = tmp_path / "FINAL.DATA"
    base_path.write_text(BASE_DATA_TEXT, encoding="utf-8")

    path_it0 = build_case(base_path, 0, {1: {"PERMX": 1.0}}, dims=(2, 2, 2), tag="it0")
    path_it1 = build_case(base_path, 0, {1: {"PERMX": 1.2}}, dims=(2, 2, 2), tag="it1")

    assert path_it0 != path_it1
    assert path_it0.exists() and path_it1.exists()


def test_run_case_substitutes_deck_placeholder(tmp_path):
    fake_deck = tmp_path / "CASE.DATA"
    fake_deck.write_text("RUNSPEC\n", encoding="utf-8")

    result = run_case(fake_deck, [sys.executable, "-c", "import sys; print(sys.argv[1])", "{deck}"])

    assert result.returncode == 0
    assert result.stdout.strip() == str(fake_deck)


# -------------------------------------------------------------- objective


def _field_rates(dates, liquid, water):
    return pd.DataFrame({"date": pd.to_datetime(dates), "liquid_rate": liquid, "water_rate": water})


def test_misfit_zero_for_identical_series():
    dates = ["2020-01-01", "2020-02-01", "2020-03-01"]
    df = _field_rates(dates, [100.0, 110.0, 120.0], [10.0, 12.0, 14.0])
    assert misfit(df, df) == pytest.approx(0.0)


def test_misfit_positive_for_different_series():
    dates = ["2020-01-01", "2020-02-01", "2020-03-01"]
    observed = _field_rates(dates, [100.0, 110.0, 120.0], [10.0, 12.0, 14.0])
    simulated = _field_rates(dates, [90.0, 130.0, 100.0], [10.0, 12.0, 14.0])
    assert misfit(observed, simulated) > 0.0


def test_misfit_raises_without_common_dates():
    observed = _field_rates(["2020-01-01"], [100.0], [10.0])
    simulated = _field_rates(["2021-01-01"], [100.0], [10.0])
    with pytest.raises(ValueError):
        misfit(observed, simulated)


def test_ensemble_misfits_sorted_ascending():
    dates = ["2020-01-01", "2020-02-01"]
    observed = _field_rates(dates, [100.0, 110.0], [10.0, 11.0])
    good = _field_rates(dates, [101.0, 109.0], [10.0, 11.0])
    bad = _field_rates(dates, [50.0, 200.0], [5.0, 20.0])

    result = ensemble_misfits(observed, {"bad": bad, "good": good})
    assert list(result.index) == ["good", "bad"]


def test_observation_vector_and_predictions_matrix_alignment():
    dates = pd.to_datetime(["2020-01-01", "2020-02-01"])
    observed = _field_rates(dates, [100.0, 200.0], [10.0, 20.0])

    obs_vector, std_vector = observation_vector(observed, list(dates), value_cols=("liquid_rate", "water_rate"))
    np.testing.assert_allclose(obs_vector, [100.0, 200.0, 10.0, 20.0])
    assert (std_vector > 0).all()

    sim_a = _field_rates(dates, [100.0, 200.0], [10.0, 20.0])
    sim_b = _field_rates(dates, [90.0, 210.0], [9.0, 22.0])
    matrix = predictions_matrix([sim_a, sim_b], list(dates), value_cols=("liquid_rate", "water_rate"))

    assert matrix.shape == (4, 2)
    np.testing.assert_allclose(matrix[:, 0], [100.0, 200.0, 10.0, 20.0])
    np.testing.assert_allclose(matrix[:, 1], [90.0, 210.0, 9.0, 22.0])


# ---------------------------------------------------------------- es_mda


def test_alpha_schedule_reciprocals_sum_to_one():
    alphas = alpha_schedule(4)
    assert len(alphas) == 4
    assert sum(1.0 / a for a in alphas) == pytest.approx(1.0)


def test_alpha_schedule_rejects_non_positive():
    with pytest.raises(ValueError):
        alpha_schedule(0)


def test_es_mda_update_shape():
    rng = np.random.default_rng(0)
    parameters = rng.normal(size=(2, 30))
    predictions = parameters + rng.normal(scale=0.1, size=(2, 30))
    observed = np.array([1.0, -1.0])
    obs_std = np.array([0.1, 0.1])

    updated = es_mda_update(parameters, predictions, observed, obs_std, alpha=4.0, rng=rng)
    assert updated.shape == parameters.shape


def test_es_mda_update_rejects_single_realization():
    parameters = np.zeros((1, 1))
    predictions = np.zeros((1, 1))
    with pytest.raises(ValueError):
        es_mda_update(parameters, predictions, np.array([0.0]), np.array([1.0]), alpha=1.0)


def test_history_matching_loop_converges_on_toy_identity_model():
    """Игрушечная модель g(m) = m: ES-MDA должен подтянуть ансамбль к observed
    и уменьшить его разброс - без всякого реального симулятора."""
    rng = np.random.default_rng(42)
    n_ens = 200
    initial = rng.normal(loc=0.0, scale=5.0, size=(2, n_ens))
    observed = np.array([5.0, -3.0])
    obs_std = np.array([0.05, 0.05])

    def forward_model(params: np.ndarray) -> np.ndarray:
        return params  # identity: predictions == parameters

    ensembles = run_history_matching_loop(initial, observed, obs_std, forward_model, n_iterations=4, seed=1)

    assert len(ensembles) == 5
    final = ensembles[-1]
    final_mean = final.mean(axis=1)
    np.testing.assert_allclose(final_mean, observed, atol=0.5)

    initial_std = initial.std(axis=1)
    final_std = final.std(axis=1)
    assert (final_std < initial_std * 0.5).all()
