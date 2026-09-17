> Производное переносимое описание для submission, 17.09.2026.
> Исторический документ не редактировался в исследовательском источнике.
> Его SHA сохранён в SOURCE_MANIFEST.json и PORTABILITY.json.

# Structured robustness: переносимое описание завершённого development-пилота

Политики main-best SAC/TQC seeds0/1/2 и classical фиксированы.
140 общих состояний = 7 групп ×20; off/on для 7 контроллеров:
98 заданий ×20 =1960 эпизодов, 980 полных пар. Timing 40 эпизодов — отдельный набор.
Состояния, единицы и точные диапазоны: [states.json](../configs/states.json),
[протокол](../../../ROBUSTNESS_PROTOCOL_20260916.md).
Nominal — новые lower-box старты; position содержит две точки на границе
training support и 18 вне него; joint выходит из совместного box по 2–4 координатам.
Отбор K-admissible состояний задаёт область применимости парного сравнения,
не доказывает решаемость swing-up или глобальную устойчивость.
Все состояния прошли admission off/on, фактических filter refusals нет.
Success 441/980→796/980; resolved превышения .24 м 585/980→0/980;
on достигает горизонта в 980/980, но 184/980 не имеет успешного final hold.
У classical известен provenance known_mismatch: SAC-подобный specification
vs фактический CourseworkCartPole-v0 без training contract. Это нельзя описывать
как полное совпадение всех environment identities; физика/R1/интегратор общие.
[Канонические таблицы и ограничения](../../../assets/structured_robustness/README.md).
Ни validation/confirmation, ни final не переопределены этим development-анализом.
