# Происхождение и область SHA

`source.json`, `imports.json`, `archives.json`, `robustness.json` — исторические
записи предыдущей подготовки/аудитов. Их SHA и текст не переписаны, устаревшие
пути не являются обещанием наличия raw-архивов в текущем Git.
`update_20260917.json` — пофайловые источники нового импорта и SHA снятых с текущего
дерева старых производных материалов. `SHA256SUMS` — **актуальный** состав submission
(кроме собственного файла). Git commit дополнительно связывает этот список.

`report_source_delivery.json`, `report_source_qa.json` сохранены как оригинальные
снимки новой редакции отчёта. Производные переносимые описания объявлены в
[PORTABILITY](../docs/report/PORTABILITY.json), исходный SOURCE_MANIFEST не переписан.
Новые wrapper scripts — packaging-адаптеры; физика, R1, фильтр, evaluator и contracts
побайтово совпадают с исследовательским источником.

Полные исходные archives/recovery, replay/runtime, verification, окружения и
личные инструкции не включены. Они остаются у автора. Ни один старый архив
или исследовательский файл при подготовке не изменялся.

[Проверки этой сборки](submission_checks.json), [точные CPU probes](inference_check.json).
