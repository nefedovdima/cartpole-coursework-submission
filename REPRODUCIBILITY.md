# Уровни воспроизводимости

Команды выполняются из корня submission. Установка — по [README](README.md).
Имеющиеся результаты не перезаписываются: новым вычислениям задавайте отдельный output.

## 1. Просмотр без вычислений

[PDF](docs/report/report.pdf), [результаты](RESULTS.md), [приложения](docs/APPENDICES.md),
[шесть видео](media/README.md). Браузер/PDF-viewer/видеоплеер достаточны.
Копии media внутри отчёта нужны относительным PDF-ссылкам; их байты идентичны корневым.
Git хранит одинаковые blobs один раз. PDF/LaTeX/CSV/рисунки приняты без научной правки.

## 2. Компактные числа и графики — чистый клон

```bash
python3 -B -m scripts.verify_submission
python3 -B -m scripts.reproduce_results --check
python3 -B -m scripts.reproduce_robustness --check
python3 -B -m cartpole.experiments.robustness_analysis --normalized results/robustness/normalized.json --check
python3 -B docs/report/scripts/check_report.py
.venv/bin/python -B -m scripts.reproduce_robustness --figures --output output-robustness
```

Первые пять команд — stdlib, без torch/Drake. Нормализованный набор проверяет
собственные identities/envelope/scientific SHA; source provenance и raw recomputation
заново не выполняются. Reporting независимо пересчитывает шесть CSV и пять PNG/SVG.
Последняя команда требует Matplotlib и нового output; существующий каталог отвергается.
Экспортные часы не влияют на научные агрегаты. README канонического пакета также
содержит историческую команду импорта review — для неё нужен внешний архив.

Main/Stage1/confirmation пересчитываются из компактных аудированных CSV:
это не независимый raw-аудит обучения. Selection/confirmation используют только
20 validation состояний, не три named-сценария. Исторические отчёты — снимки своего этапа.

Редактируемый отчёт: [инструкция](docs/report/README.md). Для сборки нужны
XeLaTeX либо Tectonic и шрифты; эти системные зависимости не входят в Python venv.
Новая сборка PDF здесь не выполнялась: поставляется точная проверенная копия.

## 3. Сохранённая политика

```bash
.venv/bin/python -B -m scripts.frozen_inference --load-check
.venv/bin/python -B -m scripts.evaluate_capsule --model SAC_seed0 --filter on
# Следующая команда действительно запускает 23 новых эпизода по 10 с:
.venv/bin/python -B -m scripts.evaluate_capsule --model SAC_seed0 --filter on --output output-sac0 --execute
```

Веса всех SAC/TQC seed0/1/2 включены, закреплены frozen bindings, metadata,
историческими manifest и SHA. Загрузка проверяет веса, библиотеки, научные исходники
и 23 сохранённые пробы на модель. Единственное уже проверенное исключение:
известная пара registry-only SHA cloud_config, только в inference-пути.
На этой подготовке проверена загрузка/probes CPU; **новые эпизоды не запускались**.
Маршрутизация evaluation проверена mock-тестом и plan. CUDA заново не проверялась.
Каждый execution создаёт fresh output, не поддерживает скрытый resume.

Классический baseline и полная checkpoint-оценка используют исходный runner:

```bash
.venv/bin/python -B -m scripts.evaluate --classical --filter on --workers 1 --output output-classical
# --execute добавляется только для сознательного запуска новых эпизодов.
```

Включена `tests/fixtures/swing_up_nominal.npz`. Её SHA закреплён в конфигурации.
Загрузка модели — десериализация доверенного артефакта; не заменяйте ZIP посторонними.
Inference-комплекты не содержат replay/runtime и **не позволяют продолжить обучение**.

## 4. Повторить обучение — отдельный дорогой эксперимент

Научный runner и frozen configs включены; старт fresh, seeds задаются явно.
Ниже пример одного исходного SAC-on, не автоматический запуск серии и не обещание
битового воспроизведения стохастического обучения. Duration — защитный предел.

```bash
.venv/bin/python -B -m scripts.train plan --config sac_training_v2_on --seed 0
.venv/bin/python -B -m scripts.train init-session --session output-training/session.json --seconds 43200 --reserve-seconds 120
.venv/bin/python -B -m cartpole.experiments.cloud_guard --session output-training/session.json --phase main --max-seconds 42000 --output-dir output-training/guard -- .venv/bin/python -B -m scripts.train run --config sac_training_v2_on --seed 0 --session output-training/session.json --max-seconds 42000 --output output-training/sac0
```

Session явно time-only, без фиктивной цены; срок не продлевается. Старые guards и
scientific runner сохранены. Plan проверен; новое обучение/guard с training в этой
подготовке не запускались. Нужен существенный запас диска для новых replay/checkpoints.
Не использовать inference ZIP как начальные веса. В main buffer_size=100000,
100000 transitions; warmup5000 ожидает 95000 updates, исходные варианты — 99872.
[Конфигурации main](configs/cloud/), [Stage1](configs/development/),
[confirmation](configs/confirmation/) — независимые исторические серии.
Для полноценного прежнего training resume нужны внешние runtime/replay и identities;
данный компактный комплект его не обещает. Final5000 не входит ни в одну команду.

## 5. Raw-аудит и повторный рендер — внешние данные

В Git нет больших review/recovery, raw-логов и исторических видеовходов.
[Исходные SHA серий](provenance/archives.json),
[robustness provenance](provenance/robustness.json),
[принятый media manifest](media/VIDEO_MANIFEST.json) сохраняют происхождение.
Для импорта robustness нужны внешние archives и их sidecar/SHA:

```bash
python3 -B -m cartpole.experiments.robustness_analysis --help
.venv/bin/python -B -m cartpole.experiments.robustness_video --help
```

Точные параметры импорта перечислены в [каноническом README](docs/assets/structured_robustness/README.md).
Видео требует также внешних historical episodes и confirmation analysis — они не
восстанавливаются из normalized metrics или моделей. На этой подготовке raw-аудит,
рендер MP4 и rollout не повторялись. Нельзя обозначать compact-check как raw-проверку.

Исторические manifests не переписаны под submission. Семь локально-зависимых
Markdown-документов отчёта заменены явно производными описаниями; исходные SHA,
причины и новые SHA — в [PORTABILITY](docs/report/PORTABILITY.json).
Актуальный состав проверяется по [SHA256SUMS](provenance/SHA256SUMS).
