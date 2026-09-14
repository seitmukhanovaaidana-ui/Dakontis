"""Ассистированная адаптация модели (history matching) поверх реального
симулятора (Eclipse/tNavigator/OPM Flow).

Этот пакет не запускает симулятор сам - он отвечает за:

- параметризацию (мультипликаторы PERMX/PERMY/PERMZ/PORO по регионам
  EQLNUM/FIPNUM) - :mod:`history_match.parameters`, :mod:`history_match.regions`;
- сборку ансамбля запускаемых кейсов - :mod:`history_match.ensemble`;
- невязку между расчётом и фактом (из `eclipse_schedule`) - :mod:`history_match.objective`;
- сам алгоритм ES-MDA - :mod:`history_match.es_mda`.

Реальный прогон симулятора и чтение его результатов (UNSMRY) - шаг,
который нужно выполнить на машине с лицензией на симулятор; в этом
пакете для него оставлены явные точки подключения (см. README).
"""
