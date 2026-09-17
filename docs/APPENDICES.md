# Электронные приложения к отчёту об исследовательском проекте

[Отчёт PDF](report/report.pdf) и [редактируемый исходник](report/report.tex).
PDF содержит одну ссылку — на главную страницу репозитория; подробные локальные
ссылки ниже работают в клоне и после распаковки полного комплекта. Адрес задан
в [настройке отчёта](report/appendix_links.tex). Репозиторий открыт
(**PUBLIC**); просмотр и скачивание материалов не требуют авторизации.

Таблица ниже связывает объекты отчёта с файлами, откуда взяты их числа.

| Объект отчёта / LaTeX label | Файл или страница |
|---|---|
| Модель, `eq:physics`, `tab:physics` | [Контракт](PROTOCOL.md), [уравнения](../cartpole/simulator/pydrake/system.py) |
| Основная серия, `app:main` | [Счётчики всех обучений](report/data/main/runs.csv), [полные компактные сводки](report/data/main/evaluation_summary.csv) |
| Выбор вариантов, `eq:config`, `app:selection` | [Точные ключи выбора настройки](report/data/stage1/selection_audit.json), [детали](report/data/stage1/detailed_comparison.csv) |
| Отрицательный confirmation | [Критерий и результат подтверждения](report/data/confirmation/confirmation_gate.json), [все исходные и новые seeds](report/data/confirmation/seeds_0_5.csv) |
| DDPG: награда и удержание | [Диагностические компоненты](report/data/confirmation/hold_component_summary.csv), [выбранная проекция траектории](report/data/confirmation/ddpg_seed5_trajectory.csv) |
| Robustness, `tab:groups`, `tab:robust_total`, `tab:interventions` | [Канонический пакет](assets/structured_robustness/README.md), [140 стартов](../configs/robustness/states.json) |
| Числа всех этапов | [RESULTS](../RESULTS.md) |
| Все рисунки отчёта | [Графики, фотография и схемы](report/figures/) |
| Самостоятельные PNG/SVG robustness | [assets](assets/structured_robustness/) |
| Приложение Г, `app:videos` | [Видео и постеры](../media/README.md), [галерея](VIDEOS.md) |
| Воспроизводимость, `app:repro` | [Уровни воспроизводимости](../REPRODUCIBILITY.md), [проверки](CHECKS.md) |
| Модели SAC и TQC | [Inference-комплекты](../models/README.md) |
| Источники и права | [Происхождение](../provenance/README.md), [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) |

Выбранные кадры и проекции траекторий — иллюстрации из прежних журналов,
а не новое измерение. Компактные сводки отсутствующие полные архивы
не заменяют.
