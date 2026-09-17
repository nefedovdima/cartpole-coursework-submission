> Производное переносимое описание для submission. Исторический SHA сохранён
> в SOURCE_MANIFEST.json; исходный исследовательский документ не изменён.

# Stage1: переносимое описание аудита

12 обучений и 12 оценок завершены. Каждый run содержит 100000 transitions;
warmup5000 — 95000 updates, lr1e4 — 99872. Для DDPG-on и SAC-off изменялся
один фактор: learning_starts=5000 либо learning_rate=1e-4.
По заранее объявленному validation-only лексикографическому score warmup5000
строго превосходит baseline у обоих методов, при minimum success=0.
Гипотеза SAC-off о росте top coverage к20k хотя бы у2/3 seeds не поддержана.
[Точные tuples](../stage1/selection_audit.json),
[все результаты](../stage1/detailed_comparison.csv),
[проверка гипотез](../stage1/hypothesis_assessment.csv).
Новая confirmation на seeds3/4/5 впоследствии не пройдена обоими вариантами;
[отдельный результат](DEVELOPMENT_CONFIRMATION_AUDIT.md).
Это переносимое описание прежнего аудита, без нового raw-пересчёта.
