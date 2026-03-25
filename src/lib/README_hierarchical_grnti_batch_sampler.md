```shell
python scripts/train/generate_hierarchical_batches.py --epochs 3 --batch-size 128 --relative-margin 0.05 --curriculum-epoch1 "0.8,0.2,0" --curriculum-epoch2 "0.6,0.3,0.1" --curriculum-epoch3plus "0.45,0.35,0.2"
```

# HierarchicalGrntiBatchSampler — README

Иерархический batch sampler для обучения би-энкодера на парах «сегмент → текст компетенции GRNTI» с контролем состава in-batch negatives. Реализация: [`hierarchical_grnti_batch_sampler.py`](hierarchical_grnti_batch_sampler.py). Подключение в обучении: `scripts/train/finetune_bi_encoder.py` (`--use-hierarchical-sampler`).

---

## 1. Назначение

Sampler совместим с **SentenceTransformerTrainer**: наследует `DefaultBatchSampler`, поддерживает **`set_epoch`** (сид эпохи, выбор curriculum).

Задача — собирать батчи под **MultipleNegativesRankingLoss / CachedMNR / CachedGISTEmbedLoss**, где негативы — **другие позитивы в батче** (второй текст пары). В отличие от `NoDuplicatesBatchSampler`, здесь явно учитываются:

- иерархия кода GRNTI (**leaf / parent / grand**);
- целевые доли типов «расстояния» между парами (**far / mid / hard**);
- защита от ложных негативов (**multi-label**, **safe-hard** по guide);
- **баланс** по редким leaf и недопредставленным разделам **grand**.

---

## 2. Входные данные (датасет)

Ожидаются колонки:

| Колонка | Смысл |
|--------|--------|
| `text1` | текст сегмента (anchor) |
| `text2` | текст компетенции для данного **leaf** (positive) |
| `doc_id` | идентификатор документа |
| `leaf` | полный leaf-код, напр. `XX.YY.ZZ` |
| `parent` | `XX.YY` |
| `grand` | `XX` |
| `doc_gold_leaves` | все leaf документа через `;` (для multi-label safety) |

Сборка таких строк — в `build_hierarchical_rows_from_segments()` в `finetune_bi_encoder.py` (двухпроходно: сначала множество leaf по `doc_id`, затем строки пар).

---

## 3. Этапы работы (по порядку)

### Этап A — инициализация (`__init__`)

1. **Метаданные по строкам** — для каждого индекса датасета создаётся `_RowMeta`: `doc_id`, `leaf`, `parent`, `grand`, `gold_leaves` (множество из `doc_gold_leaves`), тексты.
2. **Подсчёт частот** — по датасету считаются частоты **leaf** и **grand**.
3. **Базовые веса выборки `_base_w[i]`** (in-batch class balance):
   - **leaf**: \((\text{max\_leaf\_count} / \text{count}(leaf_i))^{\text{leaf\_balance\_power}}\) — редкие leaf получают больший вес;
   - **grand**: \((\text{max\_grand\_count} / \text{count}(grand_i))^{\text{grand\_balance\_weight}}\) — редкие разделы усиливаются;
   - итог: произведение двух множителей.
4. **Кэш эмбеддингов guide** — пустой словарь; при первом safe-hard для индекса считаются и кладутся в CPU-кэш эмбеддинги `text1` и `text2` (сегмент и positive).

### Этап B — начало эпохи (`__iter__`)

1. Сброс **диагностики** (`_reset_diag`).
2. **Перемешивание** индексов: `torch.randperm(n, generator=...)`.
3. **`remaining`** — словарь «ещё не отданные в батчи» индексы (порядок = случайный порядок эпохи).
4. **Curriculum** — один раз на эпоху берётся целевой вектор **(far, mid, hard)** через `_curriculum_targets()` (см. §5).

### Этап C — сборка одного батча (внутренний цикл)

Пока в батче меньше `batch_size` и есть кандидаты в `remaining`:

1. **Пул** — все индексы из `remaining`, которых ещё нет в текущем батче.
2. **Сэмплирование кандидатов для оценки** (не обходим весь пул каждый раз):
   - веса по пулу пропорциональны `_base_w[i]`;
   - из пула без возвращения берётся до **`max_scored_candidates`** индексов (`torch.multinomial`).
3. **Фильтры (жёсткие)** для каждого кандидата `j`:
   - **`_basic_ok`**: в батче ещё нет того же `doc_id`; нет того же `leaf` (один сегмент на документ + уникальный positive-код в батче).
   - **`_multilabel_ok`**: `leaf_j ∉ gold_leaves_i` и `leaf_i ∉ gold_leaves_j` для всех `i` уже в батче (ни один gold другого документа не совпадает с чужим positive как с негативом).
   - **`_safe_hard_ok`** (если задан `guide_model`): для каждой пары, где по иерархии отношение **hard** (тот же parent, разный leaf), проверяется **relative margin** по cosine similarity в guide: негатив не должен быть слишком близок к anchor относительно настоящего positive. Иначе кандидат отвергается (счётчик `reject_guide`).
4. **Scoring curriculum** — среди прошедших фильтры выбирается кандидат с лучшим `_score_candidate`: минимизируется отклонение **текущих** долей directed-рёбер (far/mid/hard) от целевого тройного вектора после **гипотетического** добавления `j` (учитываются рёбра **i↔j** в обе стороны, тип `dup` не считается).
5. **Fallback-уровни**, если лучший по score не найден:
   - при **`fallback_relaxed=True`**: случайный порядок по **всему пулу**, первый подходящий по **basic + multilabel + safe-hard** (без оптимизации curriculum) → `fallbacks++`.
   - если всё ещё пусто и есть **guide**: тот же обход пула, но **без safe-hard** (только basic + multilabel) → `unsafe_hard_relaxed++` (жёсткий негатив может оказаться «опасным» с точки зрения guide).
6. Если кандидат выбран — обновляются счётчики рёбер `cf, cm, ch` и индекс добавляется в `batch`.

Если заполнить батч до конца нельзя, внутренний цикл прерывается.

### Этап D — выдача батча и обновление `remaining`

- Если **`len(batch) == batch_size`** — батч отдаётся, диагностика батча (`_finalize_batch_diag`).
- Иначе, если батч непустой и **`drop_last=False`** — отдаётся **неполный** батч (с диагностикой).
- Если **`drop_last=True`** и батч неполный — батч **не yield**, индексы всё равно **удаляются** из `remaining` (как у `NoDuplicatesBatchSampler`: неполный хвост отбрасывается и не повторяется в той же эпохе).

После yield все индексы из `batch` удаляются из `remaining`.

### Этап E — конец эпохи

- Заполняется **`diagnostics_last_epoch`**: число батчей, суммы/доли far-mid-hard по **направленным** парам внутри батчей, счётчики отказов, fallbacks, unsafe_hard_relaxed, номер эпохи curriculum.
- При **`enable_diagnostics`** и ненулевом числе рёбер печатается строка `[hierarchical_sampler] ...`.

---

## 4. Иерархия: `relation_type` (far / mid / hard / dup)

Для пары строк (как **in-batch negative**: смотрим positive второй строки относительно первой):

| Условие | Тип |
|--------|-----|
| одинаковый `leaf` | `dup` (в графе рёбер не учитывается) |
| разный `grand` | **far** |
| один `grand`, разный `parent` | **mid** |
| один `parent`, разный `leaf` | **hard** |

В диагностике и scoring учитываются **все упорядоченные пары** `(i, j)`, `i ≠ j`, в собранном батче (соответствует тому, что в MNR каждый элемент может быть негативом для других).

---

## 5. Curriculum по эпохам

Номер эпохи **в терминах Trainer**: `training_epoch_1based = sampler.epoch + 1` (первая эпоха → `epoch=0` → используется «эпоха 1»).

| Условие | Тройка целей (far, mid, hard) |
|--------|--------------------------------|
| `epoch_1based <= 1` | `curriculum_epoch1` |
| `epoch_1based == 2` | `curriculum_epoch2` |
| иначе | `curriculum_epoch3plus` |

Строки вида `"0.8,0.2,0"` нормализуются по сумме в **`_parse_triplet`**.

**Важно:** при **`num_train_epochs=1`** используется только **`curriculum_epoch1`**; `epoch2` и `epoch3plus` не задействуются.

---

## 6. Safe-hard (guide)

Для отношения **hard** сравниваются (в **замороженной** guide-модели, L2-normalized embeddings):

- `cosine(seg_i, pos_j)` vs `cosine(seg_i, pos_i)` (и симметрично для `j` как anchor),
- отсечение, если `s_neg >= s_pos * (1 - relative_margin)`.

Если **`guide_model is None`**, проверка отключена (sampler не должен использоваться так в прод-режиме с иерархией — в `finetune_bi_encoder.py` guide задаётся при `--use-hierarchical-sampler`).

---

## 7. Фабрика для Trainer

`create_hierarchical_batch_sampler_factory(...)` парсит строки curriculum и возвращает **callable** `(dataset, batch_size, drop_last, ...) -> HierarchicalGrntiBatchSampler`, как ожидает `SentenceTransformerTrainingArguments.batch_sampler`.

---

## 8. Параметры (кратко)

| Параметр | Роль |
|----------|------|
| `guide_model` | Safe-hard + кэш эмбеддингов |
| `curriculum_epoch{1,2,3plus}` | Целевые доли far/mid/hard |
| `relative_margin` | Порог relative margin для safe-hard |
| `leaf_balance_power` | Сила оверсэмплинга редких leaf |
| `grand_balance_weight` | Сила баланса по grand |
| `max_scored_candidates` | Сколько кандидатов оценивать scoring-ом за шаг |
| `fallback_relaxed` | Включить fallback без curriculum-score |
| `enable_diagnostics` | Печать и `diagnostics_last_epoch` |

Подробнее про CLI см. `python scripts/train/finetune_bi_encoder.py --help`.

## 9. Предгенерированные батчи (ускорение обучения)

1. **Генерация** — `scripts/train/generate_hierarchical_batches.py` пишет один `.pt` с полем `batches_by_epoch` и метаданными (`dataset_size`, `args`).

2. **Обучение** — в `finetune_bi_encoder.py`:
   - `--precomputed-batches ПУТЬ.pt` (нельзя вместе с `--use-hierarchical-sampler`);
   - train-датасет собирается **так же**, как при генерации: иерархические строки, тот же `--seed` shuffle, те же `--max-train-samples` / срез;
   - `--batch-size` и **`--dataloader-drop-last`** должны совпадать с генерацией (`--drop-last` у генератора ↔ `--dataloader-drop-last` у тренера).

3. **Реализация** — [`precomputed_epoch_batch_sampler.py`](precomputed_epoch_batch_sampler.py): `PrecomputedEpochBatchSampler` + `create_precomputed_batch_sampler_factory`.

4. **Эпохи**: если `num_train_epochs` больше числа списков в файле, последний список батчей **повторяется** (с предупреждением).

Пример генерации:

```shell
python scripts/train/generate_hierarchical_batches.py --epochs 3 --batch-size 128 --relative-margin 0.05 \
  --curriculum-epoch1 "0.8,0.2,0" --curriculum-epoch2 "0.6,0.3,0.1" --curriculum-epoch3plus "0.45,0.35,0.2"
```