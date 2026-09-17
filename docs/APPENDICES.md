# Электронные приложения к отчёту об исследовательском проекте

[Отчёт PDF](report/report.pdf) и [редактируемый исходник](report/report.tex).
PDF содержит одну ссылку на главную страницу репозитория; подробные локальные
ссылки ниже работают в клоне и после распаковки полного комплекта.
Адрес закреплён в [настройке отчёта](report/appendix_links.tex).
PRIVATE-репозиторий требует авторизации и предоставленного владельцем доступа.

| Объект отчёта / LaTeX label | Файл или страница |
|---|---|
| Модель, `eq:physics`, `tab:physics` | [Контракт](PROTOCOL.md), [уравнения](../cartpole/simulator/pydrake/system.py) |
| Main, `app:main` | [Счётчики всех обучений](report/data/main/runs.csv), [полные компактные сводки](report/data/main/evaluation_summary.csv) |
| Выбор вариантов, `eq:config`, `app:selection` | [Точные ключи выбора настройки](report/data/stage1/selection_audit.json), [детали](report/data/stage1/detailed_comparison.csv) |
| Отрицательный confirmation | [Критерий и результат подтверждения](report/data/confirmation/confirmation_gate.json), [Все исходные и новые seeds](report/data/confirmation/seeds_0_5.csv) |
| DDPG: reward и удержание | [диагностические компоненты](report/data/confirmation/hold_component_summary.csv), [выбранная проекция траектории](report/data/confirmation/ddpg_seed5_trajectory.csv) |
| Robustness, `tab:groups`, `tab:robust_total`, `tab:interventions` | [Канонический пакет](assets/structured_robustness/README.md), [140 стартов](../configs/robustness/states.json) |
| Числа всех этапов | [RESULTS](../RESULTS.md) |
| Все рисунки отчёта | [Графики, фотография и схемы](report/figures/) |
| Самостоятельные PNG/SVG robustness | [assets](assets/structured_robustness/) |
| В1–В6, `app:videos` | [Видео и постеры](../media/README.md), [галерея](VIDEOS.md) |
| Воспроизводимость, `app:repro` | [Пять уровней](../REPRODUCIBILITY.md), [проверки](CHECKS.md) |
| Модели SAC/TQC | [Inference-комплекты](../models/README.md) |
| Источники и права | [provenance](../provenance/README.md), [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) |

Выбранные кадры/проекции — иллюстрации из прежних логов, не новое измерение.
Компактные сводки не заменяют отсутствующие полные архивы.
