from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import optuna
from optuna import logging as optuna_logging

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from annotation.annotator import (
    EmbeddingAnnotator,
)
from lib.eval_metrics import mean, recall_at_k
from lib.gold_io import read_gold_csv
from lib.grnti_ontology import is_leaf_grnti_code, load_ontology_code_map

ONTOLOGY_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
VALID_PATH = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_valid.csv"

BI_ENCODER_MODEL = str(
    (PROJECT_DIR / "models" / "bi-encoder-gisnauka-trainer" / "best" / "20260314_002534").resolve()
)
EMBEDDINGS_PATH = PROJECT_DIR / "data" / "ontology_grnti_embeddings_20260314_002534.npz"

THRESHOLD_RANGE = (0.4, 0.6)
TOP_K_RANGE = (5, 30)
MAX_SEGMENT_LENGTH_RANGE = (0, 800)
CONFIDENCE_AGGREGATION_CHOICES = [
    "sum",
    "sum_log_count",
    "mean_log_count",
]
FILTER_SEGMENTS_CHOICES = [True, False]

EVAL_K = 20
MAX_PRED_CODES = EVAL_K

N_TRIALS = 1000
STUDY_NAME = "annotator_hparam_search_r20"
STORAGE_URL = None

BEST_PARAMS_BASE_DIR = PROJECT_DIR / "data" / "gold" / "optuna_runs"

_VALID_ITEMS_CACHE: List = []


def _get_valid_items():
    global _VALID_ITEMS_CACHE
    if _VALID_ITEMS_CACHE:
        return _VALID_ITEMS_CACHE
    _VALID_ITEMS_CACHE = read_gold_csv(VALID_PATH)
    if not _VALID_ITEMS_CACHE:
        raise RuntimeError(f"Нет данных в valid: {VALID_PATH}")
    return _VALID_ITEMS_CACHE


def run_predictions_for_doc(
    text: str,
    annotator: EmbeddingAnnotator,
    competency_id_to_code: Dict[str, str],
    max_pred_codes: int,
) -> List[str]:
    anns = annotator.annotate(
        text=text,
        threshold=run_predictions_for_doc.threshold,
        top_k=run_predictions_for_doc.top_k,
        max_segment_length_for_context=run_predictions_for_doc.max_segment_length,
        rerank_top_k=run_predictions_for_doc.rerank_top_k,
        confidence_aggregation=run_predictions_for_doc.conf_agg,
        filter_segments=run_predictions_for_doc.filter_segments,
        use_cross_encoder_doc_score=run_predictions_for_doc.use_ce_doc_score,
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
        if not is_leaf_grnti_code(code):
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
) -> EmbeddingAnnotator:
    precomputed = embeddings_path if embeddings_path.exists() else None
    annotator = EmbeddingAnnotator(
        ontology_path=ontology_path,
        model_name=bi_model,
        precomputed_embeddings_path=precomputed,
    )
    return annotator


def objective(trial: optuna.Trial) -> float:
    threshold = trial.suggest_float("threshold", *THRESHOLD_RANGE, step=0.01)
    top_k = trial.suggest_int("top_k", *TOP_K_RANGE)
    max_segment_length = trial.suggest_int("max_segment_length_for_context", *MAX_SEGMENT_LENGTH_RANGE)
    conf_agg = trial.suggest_categorical("confidence_aggregation", CONFIDENCE_AGGREGATION_CHOICES)
    filter_segments = trial.suggest_categorical("filter_segments", FILTER_SEGMENTS_CHOICES)

    valid_items = _get_valid_items()
    annotator = build_annotator(
        ontology_path=ONTOLOGY_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        bi_model=BI_ENCODER_MODEL,
    )
    competency_id_to_code = load_ontology_code_map(ONTOLOGY_PATH)

    run_predictions_for_doc.threshold = threshold
    run_predictions_for_doc.top_k = top_k
    run_predictions_for_doc.max_segment_length = max_segment_length
    run_predictions_for_doc.rerank_top_k = 0
    run_predictions_for_doc.conf_agg = conf_agg
    run_predictions_for_doc.filter_segments = filter_segments
    run_predictions_for_doc.use_ce_doc_score = False

    recalls: List[float] = []
    for it in valid_items:
        text = it.text or ""
        if not text and it.text_path and it.text_path.exists():
            text = it.text_path.read_text(encoding="utf-8", errors="replace")
        if not text:
            continue
        pred_codes = run_predictions_for_doc(
            text=text,
            annotator=annotator,
            competency_id_to_code=competency_id_to_code,
            max_pred_codes=MAX_PRED_CODES,
        )
        r20 = recall_at_k(pred_codes, list(it.gold_codes), EVAL_K)
        recalls.append(r20)

    macro_r20 = mean(recalls)
    return macro_r20


def main() -> None:
    if not ONTOLOGY_PATH.exists():
        raise FileNotFoundError(f"Онтология не найдена: {ONTOLOGY_PATH}")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Эмбеддинги онтологии не найдены: {EMBEDDINGS_PATH}")
    if not VALID_PATH.exists():
        raise FileNotFoundError(f"Valid CSV не найден: {VALID_PATH}")

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

    from datetime import datetime

    run_dir = BEST_PARAMS_BASE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = run_dir / "best_params.json"

    initial_params = {
        "threshold": 0.52,
        "top_k": 5,
        "max_segment_length_for_context": 315,
        "confidence_aggregation": "mean_log_count",
        "filter_segments": True,
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
