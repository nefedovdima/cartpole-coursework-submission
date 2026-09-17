> Производное переносимое описание для submission, 17.09.2026.
> Исторический документ не редактировался в исследовательском источнике.
> Его SHA сохранён в SOURCE_MANIFEST.json и PORTABILITY.json.

# Основная серия: переносимое описание аудита

15 обучений завершены с 100000 transitions и 99872 updates каждое.
Полные результаты, включая неудачные DDPG/DQN/SAC-off seeds, сохранены в
[runs.csv](runs.csv) и [evaluation_summary.csv](evaluation_summary.csv).
Выбор best использует 20 validation-состояний; три named-сценария исключены
из selection. Best/last и evaluation filter off/on — отдельные строки.
Успехи best eval-on по seeds 0/1/2: SAC 20/20/20, TQC 20/20/20,
DDPG 20/0/0, DQN 20/14/20, SAC-off 0/0/0 (знаменатель 20).
Это результаты сохранённого аудита, не новая проверка raw-эпизодов в submission.
