# Шесть закреплённых inference-комплектов

SAC_seed0/1/2 и TQC_seed0/1/2 — настоящие main-best, выбранные ранее по validation20,
использованные в structured robustness; без повторного обучения и перенастройки.
[INDEX](INDEX.json) связывает SHA с [frozen bindings](../configs/robustness/bindings.json).
Каждый содержит ровно model.zip, metadata.json, probes.json, source_manifest.json.
Последний — неизменённый исторический manifest: SHA replay/runtime в нём остаются,
но эти payloads намеренно не включены. Отсутствие отмечено в INDEX; это не training checkpoint.

| Модель | Transitions best | Updates best |
|---|---:|---:|
| SAC_seed0 |70000|69872|
| SAC_seed1 |50000|49872|
| SAC_seed2 |80000|79872|
| TQC_seed0 |80000|79872|
| TQC_seed1 |40000|39872|
| TQC_seed2 |70000|69872|

Загрузчик [frozen_inference](../scripts/frozen_inference.py) сначала проверяет все
payloads и bindings, затем исходники/библиотеки, веса и сохранённые пробы действий.
Он не меняет штатную проверку полного checkpoint или training resume.
Проверены CPU load и 23 observations ×2 представления действий на каждую модель;
ни одного нового перехода среды. [Команды](../REPRODUCIBILITY.md).
