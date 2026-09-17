# Протокол экспериментов

Все серии используют [одну физическую модель](MODEL.md), горизонт 10 с и dt=0.01 с.
Классика — сохранённая номинальная траектория, TVLQR и одностороннее переключение
на верхний LQR по времени. Траектория доступна как малая
[fixture](../tests/fixtures/swing_up_nominal.npz); настройки контроллера
сохраняются в [коде](../cartpole/control/swing_up.py).

## Разделение данных

- Training: lower-box x,v∈[-0.02,0.02], theta,omega∈[-0.05,0.05], независимые RNG seeds.
- Validation: **20 фиксированных состояний**, seed resets 1000…1019, тот же узкий box.
  Участвуют в выборе best и development-конфигураций.
- Named: **3 отдельных сценария** bottom, theta=±0.02 при прочих нулях.
  Описательные результаты, исключены из selection и confirmation gate.
- Confirmation: новые **training seeds 3/4/5**, прежние validation/named состояния.
  Проверяет повторяемость обучения, а не обобщение на новые старты.
- Held-out final: не создан и не использован. Structured robustness — отдельный
  выполненный development-дизайн: [результаты](ROBUSTNESS.md),
  [исторический протокол](ROBUSTNESS_PROTOCOL_20260916.md).

Точный прежний roster: [validation_states.json](../tests/fixtures/validation_states.json).

## Успех и измерения

Верхняя область: периодическая ошибка угла относительно pi ≤10°, |omega|≤0.5,
|x|≤0.20, |v|≤0.20. Успех — полный горизонт без нарушения ограничений и нахождение
в области непрерывно последние 2 с. Это критерий **по отсчётам**.
`hold_final` — последний непрерывный интервал, `hold_longest` — максимальный;
раздельные интервалы не суммируются. Возврат R1, boundary termination и
вмешательства фильтра измеряются отдельно. Максимум |x| включает аналитический
экстремум внутри перехода; это не непрерывная проверка угла маятника.

## Основная серия и изменения одного фактора

15 обучений: SAC/TQC/DDPG/DQN filter-on × seeds0/1/2 и matched SAC-off × seeds0/1/2.
По 100000 transitions, 99872 updates. Общие baseline learning_starts=128,
batch_size=64, learning_rate=3e-4, сети 64×64, n_envs=1, buffer_size=100000,
один update после каждого eligible transition. DDPG имеет один critic и
Gaussian noise sigma=0.1; это не TD3. Полные параметры, включая различия
алгоритмов, содержатся в [configs/cloud](../configs/cloud).

Stage1: четыре варианта × seeds0/1/2: DDPG-on и SAC-off с learning_starts=5000
(**95000 updates**) либо learning_rate=1e-4 (**99872 updates**).
По отношению к baseline изменён один обучающий параметр. Остальные условия
сохранены. [Конфигурации](../configs/development) неизменны.
Confirmation: только выигравшие warmup5000, seeds3/4/5, 100000/95000,
[конфигурации](../configs/confirmation). Все seeds и неудачи учитываются.

Периодическая оценка в training filter mode на 20+3 выполняется в начале и
через 10000 transitions. Best выбирается только по полным validation-результатам:
лексикографически success_rate, hold_final, hold_longest, отрицательное время
начала подтверждённого удержания; последний tie-break — более поздний checkpoint.
Внешняя оценка best/last × evaluation off/on ×23 =92 эпизода на обучение.

Конфигурации Stage1 сравниваются по всем трём seeds: min success_rate,
mean success_rate, min mean hold_final, mean mean hold_final, mean mean hold_longest.
Требуется строгое улучшение; равенство сохраняет baseline. Только validation20,
best при evaluation в training mode. Точные tuples приведены в [RESULTS](../RESULTS.md).
Confirmation gate: каждый best success_rate≥0.8; минимум два last≥0.8.
Параметры после отбора не меняются, дополнительный бюджет автоматически не добавляется.

## Совместимость

Веса, metadata, replay/runtime payloads и probes проверяются по manifests/SHA.
Для CPU/CUDA непрерывных действий допустима только численная погрешность ≤2e-6
в нормированных единицах (≤8e-6 м/с²), rtol=0; DQN требует точного дискретного
решения. Это не послабление SHA или весов. Изменение cloud_config принимается
лишь для одной проверенной пары SHA в inference; неизвестные отличия отвергаются.
Training/resume источники остаются строгими. Не оборачивать сохранённый угол:
полные обороты в журналах существенны.
