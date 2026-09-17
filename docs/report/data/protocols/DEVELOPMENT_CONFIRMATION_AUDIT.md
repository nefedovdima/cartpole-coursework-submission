> Производное переносимое описание для submission, 17.09.2026.
> Исторический документ не редактировался в исследовательском источнике.
> Его SHA сохранён в SOURCE_MANIFEST.json и PORTABILITY.json.

# Confirmation: переносимое описание отрицательного результата

Шесть новых обучений warmup5000: DDPG-on и SAC-off, seeds 3/4/5,
100000 transitions / 95000 updates. Параметры победителей Stage 1 не менялись.
Критерий только по 20 validation-состояниям: каждый best success_rate≥.8,
не менее 2/3 last success_rate≥.8. Три named-сценария исключены.
DDPG-on best: 4/20, 0/20, 20/20; last: 0/20 у всех.
SAC-off: best и last 0/20 у всех. Обе конфигурации confirmation не прошли.
Неудачные seeds сохранены. Выбор warmup5000 на seeds 0/1/2 не означает
устойчивость к новым training seeds.
Точные числа и критерий: [seeds_0_5.csv](../confirmation/seeds_0_5.csv),
[confirmation_gate.json](../confirmation/confirmation_gate.json).
Это компактная сверка ранее проверенного аудита; raw-аудит заново не выполняется.
