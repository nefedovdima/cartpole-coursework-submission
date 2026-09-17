# Structured robustness: завершённый development-анализ

Актуальные результаты: [аудированный черновик](ROBUSTNESS_RESULTS_DRAFT.md),
[канонический компактный пакет](assets/structured_robustness/README.md).
1960 эпизодов, 980 пар, 140 стартов, SAC/TQC seeds0–2 и classical, семь групп.
Все агрегаты воспроизводятся из normalized metrics без raw/моделей/симуляции.
[Команды](../REPRODUCIBILITY.md).

[Протокол 16.09.2026](ROBUSTNESS_PROTOCOL_20260916.md) — исторический planned-снимок,
а не текущий статус завершённой серии. Старые submission-таблицы и четыре видео
заменены более поздними каноническими материалами; источник/retired SHA сохранены.
Off/on — парное вмешательство в общий controller-to-simulator путь; допускаемость
старта и семантика отказа отличаются в коде, но все 140 стартов допущены и refusal=0.
Classical `known_mismatch` сохранён. Не использовать итог как глобальную гарантию.
