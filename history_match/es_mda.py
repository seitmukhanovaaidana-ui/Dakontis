"""Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

Реализация по Emerick & Reynolds (2013), только на numpy - без внешних
пакетов вроде ERT/iterative_ensemble_smoother. Это позволяет проверить
сам алгоритм на игрушечной прямой модели (см. tests), не имея под рукой
реального симулятора: :func:`run_history_matching_loop` принимает
`forward_model` как обычную функцию (ансамбль параметров -> ансамбль
прогнозов) - в проде это обёртка над запуском Eclipse/tNavigator и
чтением UNSMRY, в тестах - любая аналитическая функция.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def alpha_schedule(n_iterations: int) -> list[float]:
    """Коэффициенты инфляции alpha_i, при которых sum(1/alpha_i) = 1 - простейший
    валидный вариант: все alpha_i равны n_iterations."""
    if n_iterations < 1:
        raise ValueError("n_iterations должно быть >= 1")
    return [float(n_iterations)] * n_iterations


def es_mda_update(
    parameters: np.ndarray,
    predictions: np.ndarray,
    observed: np.ndarray,
    obs_std: np.ndarray,
    alpha: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Один шаг ES-MDA.

    `parameters` - (n_params, n_ens), `predictions` - (n_obs, n_ens) - прогноз
    модели для текущего ансамбля параметров, `observed`/`obs_std` - (n_obs,).
    Возвращает обновлённый ансамбль параметров (n_params, n_ens).
    """
    rng = rng if rng is not None else np.random.default_rng()
    n_params, n_ens = parameters.shape
    n_obs = observed.shape[0]

    if n_ens < 2:
        raise ValueError("ES-MDA требует ансамбль минимум из 2 реализаций")
    if predictions.shape != (n_obs, n_ens):
        raise ValueError(f"predictions должен быть ({n_obs}, {n_ens}), получено {predictions.shape}")

    noise = rng.normal(size=(n_obs, n_ens)) * obs_std[:, None] * np.sqrt(alpha)
    perturbed_obs = observed[:, None] + noise

    param_anom = parameters - parameters.mean(axis=1, keepdims=True)
    pred_anom = predictions - predictions.mean(axis=1, keepdims=True)

    c_md = param_anom @ pred_anom.T / (n_ens - 1)
    c_dd = pred_anom @ pred_anom.T / (n_ens - 1)
    c_d = np.diag(obs_std**2)

    kalman_gain = c_md @ np.linalg.pinv(c_dd + alpha * c_d)
    return parameters + kalman_gain @ (perturbed_obs - predictions)


def run_history_matching_loop(
    initial_parameters: np.ndarray,
    observed: np.ndarray,
    obs_std: np.ndarray,
    forward_model: Callable[[np.ndarray], np.ndarray],
    n_iterations: int = 4,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Прогоняет ES-MDA на `n_iterations` итераций.

    На каждой итерации: `predictions = forward_model(parameters)` для
    текущего ансамбля, затем шаг :func:`es_mda_update`. Возвращает список
    ансамблей параметров длиной `n_iterations + 1` (первый элемент -
    `initial_parameters` без изменений, последний - финальный ансамбль).

    `forward_model` принимает (n_params, n_ens) и должен вернуть
    (n_obs, n_ens) - это единственная точка, куда подключается реальный
    прогон симулятора (постройка кейсов через
    :mod:`history_match.ensemble`, запуск, чтение UNSMRY и приведение к
    вектору через :func:`history_match.objective.predictions_matrix`).
    """
    rng = np.random.default_rng(seed)
    ensembles = [initial_parameters]
    for alpha in alpha_schedule(n_iterations):
        predictions = forward_model(ensembles[-1])
        ensembles.append(es_mda_update(ensembles[-1], predictions, observed, obs_std, alpha, rng=rng))
    return ensembles
