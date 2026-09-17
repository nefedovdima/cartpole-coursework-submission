# Электронные приложения к промежуточному отчёту

[PDF-кандидат](report/report.pdf) и [редактируемый исходник](report/report.tex).
Ссылки ниже относительные; URL будущего GitHub-репозитория и будущий commit не
вписаны в PDF. После публикации ссылки нужно закрепить на фактическом commit.

| Объект отчёта / LaTeX label | Файл или страница |
|---|---|
| Модель, `eq:physics`, `tab:physics` | [Контракт](PROTOCOL.md), [уравнения](../cartpole/simulator/pydrake/system.py) |
| Main, `app:main` | [runs](report/data/main/runs.csv), [полные компактные сводки](report/data/main/evaluation_summary.csv) |
| Выбор вариантов, `eq:config`, `app:selection` | [Stage1 tuples](report/data/stage1/selection_audit.json), [детали](report/data/stage1/detailed_comparison.csv) |
| Отрицательный confirmation | [gate](report/data/confirmation/confirmation_gate.json), [seeds0–5](report/data/confirmation/seeds_0_5.csv) |
| DDPG: reward и удержание | [диагностические компоненты](report/data/confirmation/hold_component_summary.csv), [выбранная проекция траектории](report/data/confirmation/ddpg_seed5_trajectory.csv) |
| Robustness, `tab:groups`, `tab:robust_total`, `tab:interventions` | [Канонический пакет](assets/structured_robustness/README.md), [140 стартов](../configs/robustness/states.json) |
| Числа всех этапов | [RESULTS](../RESULTS.md) |
| Все рисунки отчёта | [figures](report/figures/) |
| Самостоятельные PNG/SVG robustness | [assets](assets/structured_robustness/) |
| В1–В6, `app:videos` | [Видео и постеры](../media/README.md), [галерея](../README.md) |
| Воспроизводимость, `app:repro` | [Пять уровней](../REPRODUCIBILITY.md), [проверки](CHECKS.md) |
| Модели SAC/TQC | [Inference-комплекты](../models/README.md) |
| Источники и права | [provenance](../provenance/README.md), [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) |

Выбранные кадры/проекции — иллюстрации из прежних логов, не новое измерение.
Компактные сводки не заменяют отсутствующие полные архивы.
