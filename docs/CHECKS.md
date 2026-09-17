# Проверки текущего submission · 17.09.2026

Проверялся отдельный экспорт Git index в новую папку, без `.git`, исследовательской
venv и скрытых исходных данных. Зависимости установлены заново в отдельное окружение;
editable-пакет устанавливался именно из экспортированной копии. Импорты проверены
также из её родительской папки. `pip check`: No broken requirements found.

| Проверка | Фактический результат |
|---|---|
| Анализатор robustness | 56 тестов, синтетические compact/raw fixtures |
| Reporting | 29 тестов |
| Видеорендерер | 41 тест, без перекодирования принятого комплекта |
| Safety-математика | 24 теста |
| Inference compatibility | 14 тестов |
| Submission | 10 тестов |
| Inference-комплекты | 9 тестов: порча SHA, недостающие/лишние файлы, routing mock |
| Компактные результаты | 3 теста |
| Training entrypoint | 3 теста только конфигурации/отказ без guard |
| **Всего** | **189/189, OK, 116.268 с**, без пропусков |
| Main/Stage1/confirmation | Счётчики 33 обучений, validation20 tuples и отрицательные gates подтверждены compact CSV |
| Robustness | 1960/980; шесть CSV заново вычислены побайтово |
| Графики | Пять PNG + пять SVG пересозданы из normalized и побайтово совпали |
| Загрузка весов CPU | 6/6; по 23 observations, две формы запросов, всего 276 сравнений; max abs difference=0 |
| Report checks | 11 групп сверок данных/ссылок/чисел, без новой сборки PDF |
| Сохранность | 1065 исходных файлов research неизменны; 61 Python-файл cartpole идентичен источнику |
| Происхождение | Исторические manifests сохранены; семь переносимых Markdown-производных явно отмечены |
| Состав | Проверены SHA всех payloads, относительные Markdown-ссылки, AST, секреты/личные пути, размеры |

Команда целевых тестов (не полный discovery, который включает симуляционные тесты):

```bash
MPLBACKEND=Agg .venv/bin/python -B -m unittest \
 tests.test_robustness_analysis tests.test_robustness_reporting tests.test_robustness_video \
 tests.test_cart_safety tests.test_inference_compatibility tests.test_submission \
 tests.test_robustness_results tests.test_frozen_inference tests.test_training_entrypoint -v
```

[Машинная квитанция](../provenance/submission_checks.json) и
[проверка inference](../provenance/inference_check.json) не содержат локальных путей.
[Актуальный список SHA](../provenance/SHA256SUMS). Число файлов/ссылок берётся из
`python3 -B -m scripts.verify_submission`, чтобы не дублировать изменчивый счётчик.

Scientific Python-пакет и frozen configs не менялись. Только submission wrappers,
документация и проверка переносимых описаний добавлены вокруг исходных интерфейсов.
CPU: Python3.12.3, torch2.14.0+cpu, SB3/sb3-contrib2.9.0, Drake1.57.0,
NumPy2.5.3, Gymnasium1.3.0, Matplotlib3.11.2; полные pins — в requirements.
Проверка нормализованных данных дополнительно прошла с `python3 -S` (без site-packages).

**Не выполнялись:** обучение, новые policy evaluation/Drake rollout, сервер,
CUDA, final5000, полный raw-аудит, новое кодирование шести MP4, пересборка PDF.
Evaluation wrapper проверен планом и mock; его новые эпизоды не являются измеренным
результатом. Загрузка весов и вычисление сохранённых probe actions — реальные проверки.
Историческая page-by-page QA отчёта и видео перенесена как источник, а не выдана
за повторную ручную проверку. Интерактивные PDF/видеоплееры заново не проверялись.

Отсутствие лицензии upstream не устранено техническими тестами. Публикация и
окончательное авторское согласование отчёта остаются отдельными решениями.
