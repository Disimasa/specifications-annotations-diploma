from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ONTOLOGY_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
DEFAULT_EMBEDDINGS_PATH = PROJECT_DIR / "data" / "ontology_grnti_embeddings.npz"
DEFAULT_GOLD_JSONL_PATH = PROJECT_DIR / "data" / "gold" / "test_set_manual_draft.jsonl"
DEFAULT_GOLD_CSV_PATH = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test.csv"
DEFAULT_GOLD_DOC_TEST_CSV_PATH = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test_docs.csv"

SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from annotation.annotator import DEFAULT_CROSS_ENCODER_MODEL, EmbeddingAnnotator
from lib.eval_metrics import ap_at_k, mean, mrr_at_k, precision_at_k, recall_at_k
from lib.gold_io import GoldItem, read_gold_items
from lib.grnti_ontology import aggregate_codes_to_level, is_leaf_grnti_code, load_ontology_code_map
from lib.ontology_embeddings_registry import get_precomputed_embeddings_path_for_model
from lib.eval_defaults import (
    EVAL_K,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_MAX_SEGMENT_LENGTH_FOR_CONTEXT,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_CONFIDENCE_AGGREGATION,
    DEFAULT_FILTER_SEGMENTS,
)


def _default_embeddings_path_for_model(model_name: str) -> Path:
    name = Path(model_name).name or str(model_name)
    safe = name.replace(" ", "_").replace("\\", "_").replace("/", "_")
    return DEFAULT_EMBEDDINGS_PATH.with_name(f"ontology_grnti_embeddings_{safe}.npz")


BEST_MODEL_BASE = PROJECT_DIR / "models" / "bi-encoder-gisnauka-trainer" / "best"
FALLBACK_BI_ENCODER = "deepvk/USER-bge-m3"


def _resolve_best_model() -> str:
    if not BEST_MODEL_BASE.exists() or not BEST_MODEL_BASE.is_dir():
        return FALLBACK_BI_ENCODER
    subdirs = sorted(p for p in BEST_MODEL_BASE.iterdir() if p.is_dir())
    if subdirs:
        return str(subdirs[0])
    return FALLBACK_BI_ENCODER


def _list_bi_encoder_options() -> List[str]:
    """Первой опцией — best обученная модель, затем fallback и остальные модели, без родительской trainer-папки."""
    options: List[str] = [_resolve_best_model()]
    options.append(FALLBACK_BI_ENCODER)
    models_dir = PROJECT_DIR / "models"
    skip_parent_resolved = BEST_MODEL_BASE.parent.resolve()
    if models_dir.exists():
        for p in sorted(models_dir.iterdir()):
            if p.is_dir() and p.resolve() != skip_parent_resolved:
                options.append(str(p))
    seen: set[str] = set()
    uniq: List[str] = []
    for o in options:
        if o in seen:
            continue
        seen.add(o)
        uniq.append(o)
    return uniq


def _read_gold_items(gold_path: Path) -> List[GoldItem]:
    default_texts = DEFAULT_GOLD_JSONL_PATH.parent.parent / "specifications" / "texts"
    return read_gold_items(gold_path, default_texts_dir=default_texts)


def run_predictions_for_doc(
    item: GoldItem,
    annotator: EmbeddingAnnotator,
    competency_id_to_code: Dict[str, str],
    max_pred_codes: int,
) -> List[str]:
    if item.text is not None:
        text = item.text
    elif item.text_path is not None and item.text_path.exists():
        text = item.text_path.read_text(encoding="utf-8", errors="replace")
    else:
        return []
    anns = annotator.annotate(
        text=text,
        threshold=st.session_state.threshold,
        top_k=st.session_state.top_k,
        max_segment_length_for_context=st.session_state.max_segment_length_for_context,
        rerank_top_k=st.session_state.rerank_top_k,
        confidence_aggregation=st.session_state.confidence_aggregation,
        filter_segments=st.session_state.filter_segments,
        use_cross_encoder_doc_score=st.session_state.get("use_ce_doc_score", False),
    )
    codes: List[str] = []
    for ann in anns:
        cid = ann.get("competency_id")
        if not isinstance(cid, str):
            continue
        code = competency_id_to_code.get(cid)
        if not code:
            continue
        code = code.strip()
        if not is_leaf_grnti_code(code):
            continue
        if code not in codes:
            codes.append(code)
            if len(codes) >= max_pred_codes:
                break
    return codes


def evaluate(
    gold_path: Path,
    ontology_path: Path,
    emb_path: Optional[Path],
    bi_encoder_model: str,
    eval_ks: Sequence[int],
    max_pred_codes: int,
) -> Dict[str, dict]:
    # Защита от прямого выбора родительской папки trainer: мапим на best-подпапку
    try:
        if Path(bi_encoder_model).resolve() == BEST_MODEL_BASE.parent.resolve():
            bi_encoder_model = _resolve_best_model()
    except (OSError, RuntimeError):
        pass

    competency_id_to_code = load_ontology_code_map(ontology_path)
    items = _read_gold_items(gold_path)
    precomputed = emb_path if emb_path and emb_path.exists() else None

    ce_model: Optional[str] = None
    if st.session_state.rerank_top_k > 0 or st.session_state.get("use_ce_doc_score", False):
        ce_model = st.session_state.cross_encoder_model or DEFAULT_CROSS_ENCODER_MODEL

    annotator = EmbeddingAnnotator(
        ontology_path=ontology_path,
        model_name=bi_encoder_model,
        cross_encoder_model=ce_model,
        precomputed_embeddings_path=precomputed,
        use_precomputed_embeddings=bool(st.session_state.get("use_precomputed_onto_emb", True)),
    )

    per_doc: List[dict] = []
    agg_leaf: Dict[int, Dict[str, List[float]]] = {k: {"p": [], "r": [], "mrr": [], "ap": []} for k in eval_ks}
    agg_parent: Dict[int, Dict[str, List[float]]] = {k: {"p": [], "r": [], "mrr": [], "ap": []} for k in eval_ks}
    agg_grand: Dict[int, Dict[str, List[float]]] = {k: {"p": [], "r": [], "mrr": [], "ap": []} for k in eval_ks}

    total_docs = len(items)
    progress_bar = st.progress(0) if total_docs else None
    status_placeholder = st.empty()

    for idx, it in enumerate(items):
        if it.text is None and (it.text_path is None or not it.text_path.exists()):
            per_doc.append(
                {
                    "doc_id": it.doc_id,
                    "error": f"text_path не найден и text пуст: {str(it.text_path)}",
                    "gold": list(it.gold_codes),
                    "pred": [],
                }
            )
            continue

        pred_codes = run_predictions_for_doc(
            item=it,
            annotator=annotator,
            competency_id_to_code=competency_id_to_code,
            max_pred_codes=max_pred_codes,
        )

        if progress_bar is not None:
            frac = (idx + 1) / total_docs
            progress_bar.progress(int(frac * 100))
            status_placeholder.text(f"Обработано документов: {idx + 1} / {total_docs}")

        doc_metrics: Dict[str, float] = {}
        for k in eval_ks:
            p = precision_at_k(pred_codes, it.gold_codes, k)
            r = recall_at_k(pred_codes, it.gold_codes, k)
            mrr = mrr_at_k(pred_codes, it.gold_codes, k)
            ap = ap_at_k(pred_codes, it.gold_codes, k)
            doc_metrics[f"P@{k}"] = p
            doc_metrics[f"R@{k}"] = r
            doc_metrics[f"Recall@{k}"] = r
            doc_metrics[f"MRR@{k}"] = mrr
            doc_metrics[f"AP@{k}"] = ap
            agg_leaf[k]["p"].append(p)
            agg_leaf[k]["r"].append(r)
            agg_leaf[k]["mrr"].append(mrr)
            agg_leaf[k]["ap"].append(ap)

        gold_parent = aggregate_codes_to_level(it.gold_codes, 2)
        pred_parent = aggregate_codes_to_level(pred_codes, 2)
        gold_grand = aggregate_codes_to_level(it.gold_codes, 1)
        pred_grand = aggregate_codes_to_level(pred_codes, 1)

        doc_metrics_parent: Dict[str, float] = {}
        doc_metrics_grand: Dict[str, float] = {}

        for k in eval_ks:
            p = precision_at_k(pred_parent, gold_parent, k)
            r = recall_at_k(pred_parent, gold_parent, k)
            mrr = mrr_at_k(pred_parent, gold_parent, k)
            ap = ap_at_k(pred_parent, gold_parent, k)
            doc_metrics_parent[f"P@{k}"] = p
            doc_metrics_parent[f"R@{k}"] = r
            doc_metrics_parent[f"Recall@{k}"] = r
            doc_metrics_parent[f"MRR@{k}"] = mrr
            doc_metrics_parent[f"AP@{k}"] = ap
            agg_parent[k]["p"].append(p)
            agg_parent[k]["r"].append(r)
            agg_parent[k]["mrr"].append(mrr)
            agg_parent[k]["ap"].append(ap)

            p = precision_at_k(pred_grand, gold_grand, k)
            r = recall_at_k(pred_grand, gold_grand, k)
            mrr = mrr_at_k(pred_grand, gold_grand, k)
            ap = ap_at_k(pred_grand, gold_grand, k)
            doc_metrics_grand[f"P@{k}"] = p
            doc_metrics_grand[f"R@{k}"] = r
            doc_metrics_grand[f"Recall@{k}"] = r
            doc_metrics_grand[f"MRR@{k}"] = mrr
            doc_metrics_grand[f"AP@{k}"] = ap
            agg_grand[k]["p"].append(p)
            agg_grand[k]["r"].append(r)
            agg_grand[k]["mrr"].append(mrr)
            agg_grand[k]["ap"].append(ap)

        per_doc.append(
            {
                "doc_id": it.doc_id,
                "text_path": str(it.text_path),
                "gold": list(it.gold_codes),
                "pred": pred_codes,
                "metrics": doc_metrics,
                "metrics_parent": doc_metrics_parent,
                "metrics_grandparent": doc_metrics_grand,
            }
        )

    macro = {
                f"P@{k}": mean(agg_leaf[k]["p"]) for k in eval_ks
            } | {
                f"Recall@{k}": mean(agg_leaf[k]["r"]) for k in eval_ks
            } | {
                f"MRR@{k}": mean(agg_leaf[k]["mrr"]) for k in eval_ks
            } | {
                f"MAP@{k}": mean(agg_leaf[k]["ap"]) for k in eval_ks
            }

    macro_parent = {
                       f"P@{k}": mean(agg_parent[k]["p"]) for k in eval_ks
                   } | {
                       f"Recall@{k}": mean(agg_parent[k]["r"]) for k in eval_ks
                   } | {
                       f"MRR@{k}": mean(agg_parent[k]["mrr"]) for k in eval_ks
                   } | {
                       f"MAP@{k}": mean(agg_parent[k]["ap"]) for k in eval_ks
                   }

    macro_grand = {
                      f"P@{k}": mean(agg_grand[k]["p"]) for k in eval_ks
                  } | {
                      f"Recall@{k}": mean(agg_grand[k]["r"]) for k in eval_ks
                  } | {
                      f"MRR@{k}": mean(agg_grand[k]["mrr"]) for k in eval_ks
                  } | {
                      f"MAP@{k}": mean(agg_grand[k]["ap"]) for k in eval_ks
                  }

    emb_path_used = getattr(annotator, "precomputed_embeddings_path", None)

    return {
        "macro": macro,
        "macro_parent": macro_parent,
        "macro_grandparent": macro_grand,
        "per_doc": per_doc,
        "_ontology_emb_path": str(emb_path_used) if emb_path_used is not None else None,
    }


def _macro_to_table(macro: Dict[str, float], eval_ks: Sequence[int]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for k in eval_ks:
        rows.append(
            {
                "K": k,
                "P@K": macro.get(f"P@{k}", 0.0),
                "Recall@K": macro.get(f"Recall@{k}", 0.0),
                "MRR@K": macro.get(f"MRR@{k}", 0.0),
            }
        )
    return rows


def main() -> None:
    st.set_page_config(page_title="Оценка качества по GOLD", layout="centered")
    st.title("Оценка качества пайплайна по GOLD‑разметке")

    with st.sidebar:
        st.header("Источник GOLD")
        gold_source = st.radio(
            "Выбор источника",
            options=[
                "JSONL (локальные ТЗ)",
                "CSV (gisnauka выборка, старый test)",
                "CSV (gisnauka документы, новый test)",
            ],
            index=1,
        )

        st.header("Модель")
        bi_encoder_options = _list_bi_encoder_options()
        bi_encoder_model = st.selectbox(
            "Bi-encoder (SentenceTransformer)",
            options=bi_encoder_options,
            index=0,
        )

        use_ce_doc_score = st.checkbox(
            "Использовать cross-encoder для doc-score",
            value=False,
            key="use_ce_doc_score",
        )

        ce_model_default = DEFAULT_CROSS_ENCODER_MODEL
        st.session_state.cross_encoder_model = st.text_input(
            "Cross-encoder (путь или HF id)",
            value=ce_model_default,
            disabled=not use_ce_doc_score and st.session_state.get("rerank_top_k", 0) == 0,
        )

        default_emb_candidate = _default_embeddings_path_for_model(bi_encoder_model)

        st.header("Данные")
        if gold_source.startswith("JSONL"):
            gold_path_str = st.text_input("GOLD JSONL", value=str(DEFAULT_GOLD_JSONL_PATH))
        elif "документы" in gold_source:
            gold_path_str = st.text_input("GOLD CSV (docs test)", value=str(DEFAULT_GOLD_DOC_TEST_CSV_PATH))
        else:
            gold_path_str = st.text_input("GOLD CSV", value=str(DEFAULT_GOLD_CSV_PATH))

        ontology_path_str = st.text_input("Онтология", value=str(DEFAULT_ONTOLOGY_PATH))

        st.header("Параметры пайплайна")
        st.session_state.threshold = st.slider(
            "Порог score",
            0.0,
            1.0,
            DEFAULT_THRESHOLD,
            0.01,
            key="thr",
        )
        st.session_state.top_k = st.number_input(
            "Top-K сегментов на компетенцию",
            min_value=1,
            max_value=100,
            value=DEFAULT_TOP_K,
            step=1,
            key="topk",
            help="Скрипт оценки использует 50; при 10 метрики (R@20 и др.) могут отличаться.",
        )
        st.session_state.max_segment_length_for_context = st.number_input(
            "Контекст (макс. длина сегмента, 0=выкл)",
            min_value=0,
            max_value=2000,
            value=DEFAULT_MAX_SEGMENT_LENGTH_FOR_CONTEXT,
            step=10,
            key="ctxlen",
        )
        st.session_state.rerank_top_k = st.number_input(
            "Rerank top-K компетенций (0=выкл)",
            min_value=0,
            max_value=200,
            value=DEFAULT_RERANK_TOP_K,
            step=1,
            key="rerank",
        )
        agg_options = [
            "sum",
            "sum_log_count",
            "mean_log_count",
            "max",
            "mean",
            "median",
            "weighted_mean",
        ]
        try:
            agg_index_default = agg_options.index(DEFAULT_CONFIDENCE_AGGREGATION)
        except ValueError:
            agg_index_default = 0
        st.session_state.confidence_aggregation = st.selectbox(
            "Агрегация скоров сегментов",
            options=agg_options,
            index=agg_index_default,
            key="agg",
        )
        st.session_state.filter_segments = st.checkbox(
            "Фильтровать неинформативные сегменты",
            value=DEFAULT_FILTER_SEGMENTS,
            key="filt",
        )
        st.checkbox(
            "Использовать прекомпилированные эмбеддинги онтологии (если доступны)",
            value=True,
            key="use_precomputed_onto_emb",
        )

        st.header("Метрики")
        eval_ks_str = st.text_input("K для метрик (через запятую)", value="1,3,5,10,20")
        max_pred_codes = st.number_input(
            "Макс. кол-во предсказанных кодов",
            min_value=1,
            max_value=500,
            value=20,
            step=1,
        )

        run_button = st.button("Запустить оценку", type="primary")

    if not run_button:
        st.info("Выберите параметры слева и нажмите «Запустить оценку».")
        return

    gold_path = Path(gold_path_str)
    ontology_path = Path(ontology_path_str)
    # Фактический выбор NPZ идёт внутри EmbeddingAnnotator по названию модели через словарь;
    # здесь не передаём явный путь, чтобы не ломать этот маппинг.
    emb_path: Optional[Path] = None

    eval_ks: List[int] = []
    for part in eval_ks_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            eval_ks.append(int(part))
        except ValueError:
            continue
    if not eval_ks:
        eval_ks = [1, 3, 5, 10, 20]

    t0 = time.perf_counter()
    with st.spinner("Считаем метрики…"):
        summary = evaluate(
            gold_path=gold_path,
            ontology_path=ontology_path,
            emb_path=emb_path,
            bi_encoder_model=bi_encoder_model,
            eval_ks=tuple(sorted(set(eval_ks))),
            max_pred_codes=int(max_pred_codes),
        )
    elapsed_s = time.perf_counter() - t0
    mm = int(elapsed_s // 60)
    ss = int(elapsed_s % 60)
    st.caption(f"Время расчёта метрик: **{elapsed_s:.2f} сек** (≈ {mm:02d}:{ss:02d})")

    emb_path_used_str = summary.get("_ontology_emb_path")
    if emb_path_used_str:
        st.success(f"Используются прекомпилированные эмбеддинги онтологии: **{Path(emb_path_used_str).name}**")
    else:
        st.caption("Эмбеддинги онтологии вычисляются онлайн.")

    st.subheader("Листовой уровень (XX.YY.ZZ)")
    st.dataframe(_macro_to_table(summary["macro"], eval_ks), use_container_width=True)

    st.subheader("Родители (2‑й уровень, XX.YY)")
    st.dataframe(_macro_to_table(summary["macro_parent"], eval_ks), use_container_width=True)

    st.subheader("Grandparent (1‑й уровень, XX)")
    st.dataframe(_macro_to_table(summary["macro_grandparent"], eval_ks), use_container_width=True)


if __name__ == "__main__":
    main()
