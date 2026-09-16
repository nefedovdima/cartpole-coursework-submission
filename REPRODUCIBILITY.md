# Воспроизводимость

Состояние исследования: 16.09.2026. Это отдельный submission-снимок; он не
заменяет ранее замороженные серверные пакеты. Все25 файлов математического и
обучающего контракта и сохранённые конфигурации перенесены побитово.
[Источник](provenance/source.json), [пофайловые SHA](provenance/imports.json),
[идентичности исходных архивов](provenance/archives.json).
Исторические manifests не исправлялись для нового пути.

## Проверка без экспериментов

После установки из [README](README.md), из корня клона:

```bash
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -B -m scripts.verify_submission
.venv/bin/python -B -m scripts.reproduce_results --check
.venv/bin/python -B -m unittest discover -s tests -p 'test_submission.py' -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_cart_safety.py' -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_inference_compatibility.py' -v
.venv/bin/python -m pip check
git diff --check
```

Фильтр проверяется аналитическими тестами без Drake-интегрирования.
Loader-тесты используют малую искусственную модель и mocks, а не реальные
обученные checkpoints. Проверка generation восстанавливает только уже объявленный
development JSON. Ни обучение, ни rollout, ни final эти команды не запускают.
Остальные включённые тесты симулятора требуют фактических переходов; их полный
повтор при сборке submission не выполняется.

`reproduce_results` пересчитывает опубликованные числа из аудированных compact
CSV, проверяет точные правила отбора и gate. Это не повтор аудита raw episodes.
В CSV поля `source` являются именами членов архивов, а не ссылками на файлы,
которые обязаны присутствовать в клоне. Полные SHA checkpoints для каждого
best/last находятся в `results/tables/*/selected_checkpoints.csv`.

## Явная оценка сохранённого контроллера

Сначала печать плана без вычисления траекторий:

```bash
.venv/bin/python -B -m scripts.evaluate --classical --filter on --output output/classical
.venv/bin/python -B -m scripts.evaluate --checkpoint CHECKPOINT_DIRECTORY --filter on --output output/policy
```

Чтобы действительно выполнить оценку, добавьте `--execute`. Пример для классики:

```bash
.venv/bin/python -B -m scripts.evaluate --classical --filter on --workers 1 --output output/classical --execute
```

Checkpoint directory пользователь получает отдельно из проверенного recovery:
поколение выбирается по таблице SHA, не по изменяемому указателю `best`.
Нужны manifest.json, metadata.json, probes.json, model.zip, replay.pkl, runtime.pt.
Loader хеширует все payloads, загружает модель и проверяет веса, версии и действия;
replay/runtime не используются для обучения в evaluation-пути. При несовместимости
проверка останавливается. Main-model inference допускает только известную
registry-only миграцию, описанную в [протоколе](docs/PROTOCOL.md).

Оценка всегда использует исторические20 validation и3 named, горизонт10с и
прежнюю физику. Одинаковая команда с тем же output продолжает только недостающие
эпизоды после проверки receipts и источников; несовместимый output отвергается.
Это не серверная очередь и не запуск robustness. Реальное исполнение этих команд
не проверялось при сборке: новые rollout запрещены задачей подготовки.

Исходники обучающего Runner и точные конфигурации включены для исследования и
воспроизведения протокола. Старые main/development/confirmation session receipts
и серверные launchers исключены; запуск полной очереди из submission не заявляется
проверенным. Отдельный новый training запуск требует самостоятельного ограниченного
протокола и time-only session; старые weights не являются его инициализацией.

## Результаты и графики

- Main:15 обучений, seeds0/1/2;100000 transitions/99872 updates.
- Stage1:12 обучений; warmup5000:95000 updates, lr1e4:99872 updates.
- Confirmation:6 обучений, seeds3/4/5;100000/95000; обе конфигурации не подтверждены.
- Внешняя оценка: best/last × off/on ×23; named не влияет на выбор.

Таблицы, исходные графики аудитов и четыре готовых MP4 скопированы с проверкой
SHA. [Медиа-manifest](media/manifest.json) связывает видео с моделью, эпизодом и
хешем полного журнала. Полных журналов в этом клоне нет: воспроизведение анимации
из raw потребует соответствующий архив. Само видео автономно и не запускает код.

## Сборка PDF

Готовый PDF включён. Исходник LaTeX адаптирован из промежуточного отчёта:
обновлены относительные ссылки и статус уже подготовленного robustness,
добавлено раскрытие помощи генеративной модели. Исходный PDF сохранён отдельно;
его SHA приведён в provenance/source.json. Числа и прежние иллюстрации не изменены.

Для повторной сборки нужны Tectonic0.17.0 и шрифты Noto Serif, Noto Sans,
DejaVu Sans Mono с кириллицей. Они не входят в Python requirements. При первом
запуске Tectonic может загрузить TeX bundle; cache создаётся отдельно:

```bash
TECTONIC=tectonic bash scripts/build_report.sh
```

PDF может иметь другой побитовый SHA при иных шрифтах/TeX bundle; научные данные
не зависят от верстки. Результаты локальных проверок: [CHECKS](docs/CHECKS.md).

## Что не распространяется

Виртуальные окружения, кэши, переписки, внутренние инструкции, модели,
replay/runtime, полные recovery/review архивы и episode logs. Их SHA
сохранены в provenance, но доступ к raw не заменён фиктивными публичными ссылками.
Ни один файл текущего дерева не превышает50MB; Git LFS/Release для этого состава
не нужны. Если позже публиковать модели, это отдельный reviewed release с SHA.
Ограничения лицензирования и публичного размещения: [attribution](THIRD_PARTY_NOTICES.md).
