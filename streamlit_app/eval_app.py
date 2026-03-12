from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import time

import csv
import sys as _sys

import streamlit as st

# Корень проекта (на уровень выше папки streamlit_app)
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ONTOLOGY_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
DEFAULT_EMBEDDINGS_PATH = PROJECT_DIR / "data" / "ontology_grnti_embeddings.npz"
DEFAULT_GOLD_JSONL_PATH = PROJECT_DIR / "data" / "gold" / "test_set_manual_draft.jsonl"
DEFAULT_GOLD_CSV_PATH = PROJECT_DIR / "data" / "gold" / "gisnauka_samples.csv"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from annotation.annotator import DEFAULT_CROSS_ENCODER_MODEL, EmbeddingAnnotator  # noqa: E402


class GoldItem:
    """
    Универсальная запись GOLD:
    - либо локальный текстовый файл (text_path),
    - либо текст встроенный (text).
    """

    def __init__(
        self,
        doc_id: str,
        gold_codes: Tuple[str, ...],
        text_path: Optional[Path] = None,
        text: Optional[str] = None,
        top_code: Optional[str] = None,
    ) -> None:
        self.doc_id = doc_id
        self.gold_codes = gold_codes
        self.text_path = text_path
        self.text = text
        self.top_code = top_code


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ontology_code_map(ontology_path: Path) -> Dict[str, str]:
    data = _load_json(ontology_path)
    out: Dict[str, str] = {}
    for n in data.get("nodes", []):
        nid = n.get("id")
        code = n.get("code")
        if isinstance(nid, str) and isinstance(code, str) and code.strip():
            out[nid] = code.strip()
    return out


def _is_leaf_grnti_code(code: str) -> bool:
    parts = code.split(".")
    return len(parts) == 3 and all(parts) and all(p.isdigit() for p in parts)


def _to_level_code(code: str, level: int) -> Optional[str]:
    parts = code.split(".")
    if level <= 0 or level > len(parts):
        return None
    sub = parts[:level]
    if not all(p.isdigit() for p in sub):
        return None
    return ".".join(sub)


def _aggregate_codes_to_level(codes: Sequence[str], level: int) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for c in codes:
        lc = _to_level_code(c, level)
        if not lc or lc in seen:
            continue
        seen.add(lc)
        out.append(lc)
    return out


def _read_gold_items(gold_path: Path) -> List[GoldItem]:
    """
    Поддерживает два формата:
    - JSONL (локальные тексты, test_set_manual_draft.jsonl)
    - CSV (gisnauka_samples.csv), где текст = title + '\n\n' + abstract
    """
    items: List[GoldItem] = []

    if gold_path.suffix.lower() == ".csv":
        # Увеличиваем лимит размера поля, т.к. abstracts могут быть очень длинными
        try:
            csv.field_size_limit(_sys.maxsize)
        except (OverflowError, ValueError):
            # на всякий случай ставим большой, но не максимальный лимит
            csv.field_size_limit(10_000_000)

        with gold_path.open(encoding="utf-8", newline="") as f:
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
                items.append(GoldItem(doc_id=str(doc_id), gold_codes=tuple(codes), text=text, top_code=top_code))
        return items

    # JSONL по умолчанию
    for raw_line in gold_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        doc_id = str(obj.get("doc_id", "")).strip()
        if not doc_id:
            continue

        gold_codes_raw = obj.get("gold_codes") or obj.get("labels") or []
        gold_codes: List[str] = []
        if isinstance(gold_codes_raw, list):
            for x in gold_codes_raw:
                if isinstance(x, str):
                    gold_codes.append(x.strip())
                elif isinstance(x, dict) and isinstance(x.get("code"), str):
                    gold_codes.append(x["code"].strip())

        gold_codes = [c for c in gold_codes if _is_leaf_grnti_code(c)]
        gold_codes = sorted(set(gold_codes))

        tp = obj.get("text_path")
        if isinstance(tp, str) and tp.strip():
            text_path = Path(tp)
        else:
            text_path = DEFAULT_GOLD_JSONL_PATH.parent.parent / "specifications" / "texts" / f"{doc_id}.txt"

        items.append(GoldItem(doc_id=doc_id, gold_codes=tuple(gold_codes), text_path=text_path))
    return items


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


def mrr_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = set(gold)
    if not g:
        return 1.0
    for i, p in enumerate(pred[:k], start=1):
        if p in g:
            return 1.0 / float(i)
    return 0.0


def ap_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = set(gold)
    if not g:
        return 1.0
    hits = 0
    s = 0.0
    for i, p in enumerate(pred[:k], start=1):
        if p in g:
            hits += 1
            s += hits / float(i)
    return s / float(len(g))


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / float(len(xs)) if xs else 0.0


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
        if not _is_leaf_grnti_code(code):
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
    competency_id_to_code = _load_ontology_code_map(ontology_path)
    items = _read_gold_items(gold_path)

    # Всегда используем ту же выборку, что и в Optuna (только для CSV):
    # - группируем по top_code (XX)
    # - для каждого top_code берём ПЕРВЫЙ документ
    # Совпадает с логикой в tools/tune_hyperparams_optuna.py.
    if gold_path.suffix.lower() == ".csv":
        by_top: Dict[str, List[GoldItem]] = {}
        for it in items:
            top = (it.top_code or (it.gold_codes[0].split(".")[0] if it.gold_codes else "")).strip()
            if not top:
                continue
            by_top.setdefault(top, []).append(it)
        reduced: List[GoldItem] = []
        for top in sorted(by_top.keys()):
            docs = by_top[top]
            if docs:
                reduced.append(docs[0])
        items = reduced
    precomputed = emb_path if emb_path and emb_path.exists() else None

    ce_model: Optional[str] = None
    if st.session_state.rerank_top_k > 0 or st.session_state.get("use_ce_doc_score", False):
        ce_model = st.session_state.cross_encoder_model or DEFAULT_CROSS_ENCODER_MODEL

    annotator = EmbeddingAnnotator(
        ontology_path=ontology_path,
        model_name=bi_encoder_model,
        cross_encoder_model=ce_model,
        precomputed_embeddings_path=precomputed,
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

        gold_parent = _aggregate_codes_to_level(it.gold_codes, 2)
        pred_parent = _aggregate_codes_to_level(pred_codes, 2)
        gold_grand = _aggregate_codes_to_level(it.gold_codes, 1)
        pred_grand = _aggregate_codes_to_level(pred_codes, 1)

        doc_metrics_parent: Dict[str, float] = {}
        doc_metrics_grand: Dict[str, float] = {}

        for k in eval_ks:
            # parent
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

            # grandparent
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

    return {
        "macro": macro,
        "macro_parent": macro_parent,
        "macro_grandparent": macro_grand,
        "per_doc": per_doc,
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
            options=["JSONL (локальные ТЗ)", "CSV (gisnauka выборка)"],
            index=1,
        )

        st.header("Данные")
        if gold_source.startswith("JSONL"):
            gold_path_str = st.text_input("GOLD JSONL", value=str(DEFAULT_GOLD_JSONL_PATH))
        else:
            gold_path_str = st.text_input("GOLD CSV", value=str(DEFAULT_GOLD_CSV_PATH))

        ontology_path_str = st.text_input("Онтология", value=str(DEFAULT_ONTOLOGY_PATH))
        emb_path_str = st.text_input("Эмбеддинги онтологии (.npz)", value=str(DEFAULT_EMBEDDINGS_PATH))

        st.header("Параметры пайплайна")
        st.session_state.threshold = st.slider("Порог score", 0.0, 1.0, 0.55, 0.01, key="thr")
        st.session_state.top_k = st.number_input(
            "Top-K сегментов на компетенцию",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="topk",
        )
        st.session_state.max_segment_length_for_context = st.number_input(
            "Контекст (макс. длина сегмента, 0=выкл)",
            min_value=0,
            max_value=2000,
            value=0,
            step=10,
            key="ctxlen",
        )
        st.session_state.rerank_top_k = st.number_input(
            "Rerank top-K компетенций (0=выкл)",
            min_value=0,
            max_value=200,
            value=0,
            step=1,
            key="rerank",
        )
        st.session_state.confidence_aggregation = st.selectbox(
            "Агрегация скоров сегментов",
            options=[
                "sum",
                "sum_log_count",
                "mean_log_count",
                "max",
                "mean",
                "median",
                "weighted_mean",
            ],
            index=0,
            key="agg",
        )
        st.session_state.filter_segments = st.checkbox("Фильтровать неинформативные сегменты", value=True, key="filt")

        st.header("Метрики")
        eval_ks_str = st.text_input("K для метрик (через запятую)", value="1,3,5,10,20")
        max_pred_codes = st.number_input(
            "Макс. кол-во предсказанных кодов",
            min_value=1,
            max_value=500,
            value=20,
            step=1,
        )

        st.header("Модель")
        bi_encoder_model = st.text_input(
            "Bi-encoder (SentenceTransformer)",
            value="deepvk/USER-bge-m3",
        )

        use_ce_doc_score = st.checkbox(
            "Использовать cross-encoder для doc-score",
            value=False,
            key="use_ce_doc_score_cb",
        )

        ce_model_default = DEFAULT_CROSS_ENCODER_MODEL
        st.session_state.cross_encoder_model = st.text_input(
            "Cross-encoder (путь или HF id)",
            value=ce_model_default,
            disabled=not use_ce_doc_score and st.session_state.rerank_top_k == 0,
        )

        run_button = st.button("Запустить оценку", type="primary")

    if not run_button:
        st.info("Выберите параметры слева и нажмите «Запустить оценку».")
        return

    gold_path = Path(gold_path_str)
    ontology_path = Path(ontology_path_str)
    emb_path = Path(emb_path_str) if emb_path_str.strip() else None

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

    st.subheader("Листовой уровень (XX.YY.ZZ)")
    st.dataframe(_macro_to_table(summary["macro"], eval_ks), use_container_width=True)

    st.subheader("Родители (2‑й уровень, XX.YY)")
    st.dataframe(_macro_to_table(summary["macro_parent"], eval_ks), use_container_width=True)

    st.subheader("Grandparent (1‑й уровень, XX)")
    st.dataframe(_macro_to_table(summary["macro_grandparent"], eval_ks), use_container_width=True)


if __name__ == "__main__":
    main()

