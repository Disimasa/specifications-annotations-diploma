# Diploma: Семантическая аннотация ТЗ по онтологии ГРНТИ

Репозиторий содержит пайплайн для аннотации текстов технических заданий (ТЗ) по онтологии компетенций ГРНТИ:
- сегментация текста;
- поиск релевантных компетенций bi-encoder моделью;
- опциональный re-ranking cross-encoder моделью;
- агрегация результатов по документу;
- оценка качества на GOLD-разметке.

Основной интерфейс для инференса в проде: `apps/app.py` (Streamlit).

## Что делает проект

- Принимает входной текст ТЗ (`.txt` и `.docx`).
- Разбивает текст на сегменты.
- Сопоставляет сегменты с узлами онтологии на основе эмбеддингов.
- Строит итоговый top-N компетенций с подробностями по сегментам.
- Позволяет запускать оценку качества на размеченных датасетах.

## Структура репозитория

- `apps/app.py` — основное Streamlit-приложение для аннотации.
- `apps/eval_app.py` — Streamlit-интерфейс для оценки качества.
- `src/annotation` — ядро инференса (`EmbeddingAnnotator`, сегментация, фильтрация).
- `src/lib` — утилиты (метрики, загрузка данных, онтология, маппинг precomputed-эмбеддингов).
- `scripts/train` — скрипты обучения/подготовки данных.
- `scripts/eval/evaluate_r20_full_pipeline.py` — CLI-оценка полного пайплайна.
- `data` — онтология, GOLD-наборы, предрассчитанные эмбеддинги.
- `models` — обученные и финальные модели.

## Модель и данные для прод-инференса

Текущий прод-конфиг зафиксирован на:
- модели: `models/final`
- онтологии: `data/ontology_grnti_with_llm.json`
- предрассчитанных эмбеддингах: `data/ontology_grnti_embeddings_fnfilter001.npz`

Папка `models/final` содержит только файлы, необходимые для инференса `SentenceTransformer` (без тренировочных чекпоинтов и оптимизатора).

## Быстрый запуск локально

```bash
pip install -r apps/requirements.txt
streamlit run apps/app.py
```

Приложение поднимется на `http://localhost:8501`.

## Прод-запуск в Docker

```bash
docker compose up -d --build
```
## Требуемые ресурсы

### Размер обязательных прод-артефактов

- `models/final`: **1,372.83 MB**
- `data/ontology_grnti_with_llm.json`: **8.58 MB**
- `data/ontology_grnti_embeddings_fnfilter001.npz`: **29.18 MB**

Итого обязательные mounted-файлы: **1,410.59 MB** (~**1.38 GB**).

### Оценка требований к машине

- **Disk (минимум):**
  - обязательные артефакты: ~1.38 GB
  - \+ зависимость
  - \+ Docker image
  - итого 6.36GB GB

- **RAM:**
  - 8 - 16гб?

## Сравнение онтологий (обучение)

Скрипт: `scripts/train/run_base_model_training_pipeline.py`

Предбатчи **общие** для всех онтологий (зависят от CSV сегментов и кодов ГРНТИ, не от текста описаний). При смене онтологии пересчитываются только эмбеддинги, обучение и оценка.

```bash
# Полная выборка; предбатчи: hb_grandfocus1-0-0_...pt (curriculum 1,0,0)
python scripts/train/run_base_model_training_pipeline.py --ontology data/my_ontology_v1.json

# Другая онтология — те же батчи, новая модель и метрики
python scripts/train/run_base_model_training_pipeline.py \
  --ontology data/my_ontology_v2.json \
  --trained-model-path models/ontology-v2

# Smoke-тест (полный train + предбатчи + FN; первые 4 батча)
python scripts/train/run_base_model_training_pipeline.py --ontology data/my_ontology.json --max-batches 4 --force
```

Аргументы:
- `--ontology` — JSON онтологии для обучения и оценки;
- `--max-batches` — smoke: первые N предбатчей (полный train, FN-фильтр, как prod);
- `--max-train-samples` — устаревший alias: ceil(N/128) батчей при предбатчах;
- `--trained-model-path` — каталог модели;
- `--precomputed-batches` — свой `.pt` (по умолчанию `hb_grandfocus1-0-0_...pt` на полной выборке);
- `--regenerate-batches` — пересобрать общие предбатчи (редко нужно);
- `--force` — переобучить эту онтологию заново.

Артефакты по онтологии: `data/pipeline/ontology_runs/{run_key}/`. Состояние: `data/pipeline/base_model_state.json`.
