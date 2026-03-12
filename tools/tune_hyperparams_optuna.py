from __future__ import annotations

"""
Подбор гиперпараметров пайплайна аннотации через Optuna.

Целевая метрика: Recall@20 для листовых нод (кодов ГРНТИ формата XX.YY.ZZ),
которую мы максимизируем.

Высчитываем метрику на небольшой подвыборке (N_SAMPLE_DOCS документов)
из GOLD-разметки (по умолчанию — CSV выборка с gisnauka).

ВАЖНО:
- Границы гиперпараметров заданы в константах ниже и легко меняются.
- Скрипт не трогает Streamlit, это отдельный оффлайн-подбор.
"""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import optuna
from optuna import logging as optuna_logging


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from annotation.annotator import (  # noqa: E402
    DEFAULT_CROSS_ENCODER_MODEL,
    EmbeddingAnnotator,
)


# =========================
# КОНСТАНТЫ / ПАРАМЕТРЫ
# =========================

# Пути к данным/моделям
ONTOLOGY_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
EMBEDDINGS_PATH = PROJECT_DIR / "data" / "ontology_grnti_embeddings.npz"
GOLD_PATH = PROJECT_DIR / "data" / "gold" / "gisnauka_samples.csv"

BI_ENCODER_MODEL = "deepvk/USER-bge-m3"

# Варианты cross-encoder моделей:
# - "hf_dity"      : публичная DiTy/cross-encoder-russian-msmarco (HuggingFace)
# - "local_rusbeir": твоя модель на RusBEIR
# - "local_sts"    : твоя модель на STS
CE_MODEL_HF_DITY = DEFAULT_CROSS_ENCODER_MODEL  # DiTy/cross-encoder-russian-msmarco
CE_MODEL_RUSBEIR = "models/cross-encoder-rusbeir/final_model"

# Доступные варианты cross-encoder: публичная DiTy и (опционально) локальная RusBEIR
CROSS_ENCODER_MODEL_MAP: Dict[str, str] = {"hf_dity": CE_MODEL_HF_DITY}
if (PROJECT_DIR / CE_MODEL_RUSBEIR).exists():
    CROSS_ENCODER_MODEL_MAP["local_rusbeir"] = CE_MODEL_RUSBEIR

CROSS_ENCODER_MODEL_CHOICES = list(CROSS_ENCODER_MODEL_MAP.keys())

# Hyperparameter search space
THRESHOLD_RANGE = (0.3, 0.9)  # (min, max); первая точка будет 0.55
TOP_K_RANGE = (5, 30)  # кол-во сегментов на компетенцию
MAX_SEGMENT_LENGTH_RANGE = (0, 800)  # 0 = без контекста
RERANK_TOP_K_RANGE = (0, 30)  # 0 = без cross-encoder
CONFIDENCE_AGGREGATION_CHOICES = [
    "sum",
    "sum_log_count",
    "mean_log_count",
]
FILTER_SEGMENTS_CHOICES = [True, False]
USE_CE_DOC_SCORE_CHOICES = [False, True]

# Метрики считаем только по K=20 для листьев
EVAL_K = 20
MAX_PRED_CODES = EVAL_K  # ограничиваем длину списка предсказаний

# Настройки Optuna
N_TRIALS = 100
STUDY_NAME = "annotator_hparam_search_r20"
STORAGE_URL = None  # например: "sqlite:///optuna_study.db" или None для in-memory

# Базовая директория для сохранения лучших параметров по датам
BEST_PARAMS_BASE_DIR = PROJECT_DIR / "data" / "gold" / "optuna_runs"


# =========================
# Утилиты
# =========================


def _is_leaf_grnti_code(code: str) -> bool:
    parts = code.split(".")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


@dataclass
class GoldItem:
    doc_id: str
    text: str
    gold_codes: Tuple[str, ...]
    top_code: Optional[str] = None


def _read_gold_csv(path: Path) -> List[GoldItem]:
    items: List[GoldItem] = []

    # Увеличиваем лимит размера поля, т.к. abstracts могут быть очень длинными
    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            codes_raw = (row.get("grnti_codes") or "").strip()
            if not codes_raw:
                continue
            codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
            codes = [c for c in codes if _is_leaf_grnti_code(c)]
            codes = sorted(set(codes))
            if not codes:
                continue
            text = f"{title}\n\n{abstract}".strip()
            if not text:
                continue
            doc_id = row.get("doc_id") or f"gisnauka_{i}"
            top_code = (row.get("top_code") or "").strip() or codes[0].split(".")[0]
            items.append(GoldItem(doc_id=str(doc_id), text=text, gold_codes=tuple(codes), top_code=top_code))
    return items


# Глобальная фиксированная подвыборка: по одному документу на каждую верхнеуровневую рубрику (XX)
_SAMPLE_ITEMS: List[GoldItem] = []


def _get_sample_items() -> List[GoldItem]:
    """
    Строим детерминированную выборку один раз:
    - читаем все строки CSV;
    - группируем по top_code (XX);
    - для каждого top_code берём ПЕРВЫЙ документ;
    - итоговый список фиксирован и не меняется между запусками trial'ов.
    """
    global _SAMPLE_ITEMS
    if _SAMPLE_ITEMS:
        return _SAMPLE_ITEMS

    all_items = _read_gold_csv(GOLD_PATH)
    if not all_items:
        raise RuntimeError(f"Нет данных в GOLD: {GOLD_PATH}")

    by_top: Dict[str, List[GoldItem]] = {}
    for it in all_items:
        top = (it.top_code or (it.gold_codes[0].split(".")[0] if it.gold_codes else "")).strip()
        if not top:
            continue
        by_top.setdefault(top, []).append(it)

    sample: List[GoldItem] = []
    for top in sorted(by_top.keys()):
        docs = by_top[top]
        if docs:
            sample.append(docs[0])

    if not sample:
        raise RuntimeError("Не удалось собрать фиксированную подвыборку документов по верхнеуровневым рубрикам.")

    _SAMPLE_ITEMS = sample
    return _SAMPLE_ITEMS


def precision_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    pred_k = pred[:k]
    if not pred_k:
        return 0.0
    g = set(gold)
    return sum(1 for p in pred_k if p in g) / float(len(pred_k))


def recall_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    g = set(gold)
    if not g:
        return 1.0
    pred_k = pred[:k]
    return sum(1 for p in pred_k if p in g) / float(len(g))


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / float(len(xs)) if xs else 0.0


def run_predictions_for_doc(
    text: str,
    annotator: EmbeddingAnnotator,
    competency_id_to_code: Dict[str, str],
    max_pred_codes: int,
) -> List[str]:
    anns = annotator.annotate(
        text=text,
        threshold=run_predictions_for_doc.threshold,  # type: ignore[attr-defined]
        top_k=run_predictions_for_doc.top_k,  # type: ignore[attr-defined]
        max_segment_length_for_context=run_predictions_for_doc.max_segment_length,  # type: ignore[attr-defined]
        rerank_top_k=run_predictions_for_doc.rerank_top_k,  # type: ignore[attr-defined]
        confidence_aggregation=run_predictions_for_doc.conf_agg,  # type: ignore[attr-defined]
        filter_segments=run_predictions_for_doc.filter_segments,  # type: ignore[attr-defined]
        use_cross_encoder_doc_score=run_predictions_for_doc.use_ce_doc_score,  # type: ignore[attr-defined]
    )
    codes: List[str] = []
    for ann in anns:
        cid = ann.get("competency_id")
        if not isinstance(cid, str):
            continue
        code = competency_id_to_code.get(cid)
        if not isinstance(code, str):
            continue
        code = code.strip()
        if not _is_leaf_grnti_code(code):
            continue
        if code not in codes:
            codes.append(code)
            if len(codes) >= max_pred_codes:
                break
    return codes


def build_annotator(
    ontology_path: Path,
    embeddings_path: Path,
    bi_model: str,
    ce_model: Optional[str],
) -> EmbeddingAnnotator:
    precomputed = embeddings_path if embeddings_path.exists() else None
    annotator = EmbeddingAnnotator(
        ontology_path=ontology_path,
        model_name=bi_model,
        cross_encoder_model=ce_model,
        precomputed_embeddings_path=precomputed,
    )
    return annotator


def objective(trial: optuna.Trial) -> float:
    # --- выбор гиперпараметров ---
    threshold = trial.suggest_float("threshold", *THRESHOLD_RANGE)
    top_k = trial.suggest_int("top_k", *TOP_K_RANGE)
    max_segment_length = trial.suggest_int("max_segment_length_for_context", *MAX_SEGMENT_LENGTH_RANGE)
    rerank_top_k = trial.suggest_int("rerank_top_k", *RERANK_TOP_K_RANGE)
    conf_agg = trial.suggest_categorical("confidence_aggregation", CONFIDENCE_AGGREGATION_CHOICES)
    filter_segments = trial.suggest_categorical("filter_segments", FILTER_SEGMENTS_CHOICES)
    use_ce_doc_score = trial.suggest_categorical("use_ce_doc_score", USE_CE_DOC_SCORE_CHOICES)

    # Если rerank_top_k = 0, cross-encoder по сути не используется,
    # и флаг use_ce_doc_score не имеет смысла.
    if rerank_top_k == 0:
        # Cross-encoder вообще не используется: не трогаем модель и игнорируем use_ce_doc_score
        ce_model = None
        use_ce_doc_score = False
    else:
        # Только когда реально есть re-ranking, подбираем cross_encoder_model
        ce_model_key = trial.suggest_categorical("cross_encoder_model", CROSS_ENCODER_MODEL_CHOICES)
        ce_model = CROSS_ENCODER_MODEL_MAP[ce_model_key]

    # --- подготовка данных ---
    sample = _get_sample_items()

    # --- создаём аннотатор с выбранной моделью cross-encoder ---
    annotator = build_annotator(
        ontology_path=ONTOLOGY_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        bi_model=BI_ENCODER_MODEL,
        ce_model=ce_model,
    )

    # мапа competency_id -> code из онтологии
    onto = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    competency_id_to_code: Dict[str, str] = {}
    for n in onto.get("nodes", []):
        nid = n.get("id")
        code = n.get("code")
        if isinstance(nid, str) and isinstance(code, str) and code.strip():
            competency_id_to_code[nid] = code.strip()

    # "прокидываем" текущие гиперпараметры в функцию аннотации документов
    run_predictions_for_doc.threshold = threshold  # type: ignore[attr-defined]
    run_predictions_for_doc.top_k = top_k  # type: ignore[attr-defined]
    run_predictions_for_doc.max_segment_length = max_segment_length  # type: ignore[attr-defined]
    run_predictions_for_doc.rerank_top_k = rerank_top_k  # type: ignore[attr-defined]
    run_predictions_for_doc.conf_agg = conf_agg  # type: ignore[attr-defined]
    run_predictions_for_doc.filter_segments = filter_segments  # type: ignore[attr-defined]
    run_predictions_for_doc.use_ce_doc_score = use_ce_doc_score  # type: ignore[attr-defined]

    recalls: List[float] = []

    for it in sample:
        pred_codes = run_predictions_for_doc(
            text=it.text,
            annotator=annotator,
            competency_id_to_code=competency_id_to_code,
            max_pred_codes=MAX_PRED_CODES,
        )
        r20 = recall_at_k(pred_codes, it.gold_codes, EVAL_K)
        recalls.append(r20)

    macro_r20 = mean(recalls)
    return macro_r20


def main() -> None:
    if not ONTOLOGY_PATH.exists():
        raise FileNotFoundError(f"Онтология не найдена: {ONTOLOGY_PATH}")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Эмбеддинги онтологии не найдены: {EMBEDDINGS_PATH}")
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"GOLD CSV не найден: {GOLD_PATH}")

    # Включаем подробный лог Optuna (каждая итерация + сводка best trial)
    optuna_logging.set_verbosity(optuna_logging.INFO)

    sampler = optuna.samplers.TPESampler()

    if STORAGE_URL:
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=STORAGE_URL,
            direction="maximize",
            sampler=sampler,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    # Директория для этого запуска (дата-время)
    from datetime import datetime

    run_dir = BEST_PARAMS_BASE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = run_dir / "best_params.json"

    # Первая точка — hand-crafted baseline с threshold=0.55 и дефолтными параметрами
    initial_params = {
        "threshold": 0.55,
        "top_k": 10,
        "max_segment_length_for_context": 0,
        "rerank_top_k": 0,
        "confidence_aggregation": "sum",
        "filter_segments": True,
        "use_ce_doc_score": False,
        "cross_encoder_model": CROSS_ENCODER_MODEL_CHOICES[0],  # hf_dity
    }
    study.enqueue_trial(initial_params)

    def log_trial(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        logger = optuna_logging.get_logger("optuna")
        best = study.best_trial
        logger.info(
            "Trial %d finished with value=%.4f, params=%s. Best so far: trial %d with value=%.4f.",
            trial.number,
            trial.value,
            trial.params,
            best.number,
            best.value,
        )
        # сохраняем лучшие параметры в файл после КАЖДОГО trial
        best_payload = {
            "value": best.value,
            "params": best.params,
            "number": best.number,
        }
        best_params_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_trial])

    print("Лучший trial:")
    print(f"  value (Recall@{EVAL_K}): {study.best_value:.4f}")
    print("  params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()

