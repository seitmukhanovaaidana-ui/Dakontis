"""Пространство параметров адаптации: мультипликаторы свойств по регионам.

Каждый настраиваемый параметр - это пара (номер региона, свойство),
например (2, 'PERMX'). ES-MDA работает с параметрами как с вектором
чисел без знака размерности, поэтому здесь же - упаковка/распаковка
между "человеческим" представлением (dict региона в dict свойств) и
numpy-вектором, а также сэмплирование начального ансамбля.

Мультипликаторы всегда положительны (это коэффициент, а не абсолютное
значение), поэтому ES-MDA ведётся в log-пространстве - `x = ln(k)` -
это гарантирует, что после апдейта параметр не станет отрицательным
или нулевым (после `exp(x)` он всегда положителен), и что априорное
распределение "во сколько раз может быть больше/меньше" симметрично.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterSpace:
    """Фиксированный порядок параметров (регион, свойство) для ансамбля."""

    region_ids: tuple[int, ...]
    properties: tuple[str, ...]

    @property
    def keys(self) -> list[tuple[int, str]]:
        """Список (регион, свойство) в фиксированном порядке - строки вектора параметров."""
        return [(region, prop) for region in self.region_ids for prop in self.properties]

    @property
    def n_params(self) -> int:
        return len(self.region_ids) * len(self.properties)

    def sample_prior_log(
        self, n_realizations: int, low: float, high: float, seed: int | None = None
    ) -> np.ndarray:
        """Начальный ансамбль в log-пространстве: log-равномерно на [low, high].

        `low`/`high` - границы самого мультипликатора (например 0.3 и 3.0
        означают "от -70% до +200% от базового значения"), не логарифма.
        Возвращает матрицу (n_params, n_realizations).
        """
        if low <= 0 or high <= 0 or low >= high:
            raise ValueError("Нужно 0 < low < high (границы самого мультипликатора)")
        rng = np.random.default_rng(seed)
        log_low, log_high = np.log(low), np.log(high)
        return rng.uniform(log_low, log_high, size=(self.n_params, n_realizations))

    def to_multiplier_dict(self, log_vector: np.ndarray) -> dict[int, dict[str, float]]:
        """Один столбец ансамбля (вектор длины n_params, в log-пространстве) -> dict{регион: {свойство: k}}."""
        if len(log_vector) != self.n_params:
            raise ValueError(f"Ожидался вектор длины {self.n_params}, получено {len(log_vector)}")
        result: dict[int, dict[str, float]] = {}
        for (region, prop), log_value in zip(self.keys, log_vector):
            result.setdefault(region, {})[prop] = float(np.exp(log_value))
        return result

    def from_multiplier_dict(self, multipliers: dict[int, dict[str, float]]) -> np.ndarray:
        """Обратное преобразование: dict{регион: {свойство: k}} -> вектор log(k) в фиксированном порядке."""
        return np.array([np.log(multipliers[region][prop]) for region, prop in self.keys])
