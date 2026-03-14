from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from annotation.annotator import EmbeddingAnnotator
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from lib.eval_metrics import mrr_at_k, precision_at_k, recall_at_k
from lib.gold_io import GoldItem, read_gold_csv, read_gold_jsonl
from lib.grnti_ontology import (
    is_leaf_grnti_code,
    load_ontology_code_map,
    load_ontology_texts,
)

DEFAULT_ONTOLOGY = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
DEFAULT_TEST_GISNAUKA = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test.csv"
DEFAULT_TEST_GISNAUKA_DOCS = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test_docs.csv"
DEFAULT_GOLD_JSONL = PROJECT_DIR / "data" / "gold" / "test_set_manual_draft.jsonl"
DEFAULT_VALID_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_valid.csv"

BEST_MODEL_BASE = PROJECT_DIR / "models" / "bi-encoder-gisnauka-trainer" / "best"
FALLBACK_MODEL = "deepvk/USER-bge-m3"

EVAL_K = 20
DEFAULT_THRESHOLD = 0.55
DEFAULT_TOP_K = 50
DEFAULT_MAX_SEGMENT_LENGTH_FOR_CONTEXT = 0
DEFAULT_RERANK_TOP_K = 0
DEFAULT_CONFIDENCE_AGGREGATION = "sum"
DEFAULT_FILTER_SEGMENTS = True


def _resolve_default_model() -> str:
    if not BEST_MODEL_BASE.exists() or not BEST_MODEL_BASE.is_dir():
        return FALLBACK_MODEL
    subdirs = sorted(p for p in BEST_MODEL_BASE.iterdir() if p.is_dir())
    if subdirs:
        return str(subdirs[0])
    return FALLBACK_MODEL


def _annotations_to_codes(
    annotations: List[dict],
    competency_id_to_code: Dict[str, str],
    max_codes: int,
) -> List[str]:
    codes: List[str] = []
    for ann in annotations:
        cid = ann.get("competency_id")
        if not isinstance(cid, str):
            continue
        code = competency_id_to_code.get(cid)
        if not code or not is_leaf_grnti_code(code):
            continue
        if code not in codes:
            codes.append(code)
            if len(codes) >= max_codes:
                break
    return codes


def _retrieve_codes_to_list(
    id_scores: List[Tuple[str, float]],
    competency_id_to_code: Dict[str, str],
    max_codes: int,
) -> List[str]:
    codes: List[str] = []
    for cid, _ in id_scores:
        code = competency_id_to_code.get(cid)
        if not code or not is_leaf_grnti_code(code):
            continue
        if code not in codes:
            codes.append(code)
            if len(codes) >= max_codes:
                break
    return codes


def evaluate_doc_level(
    model: SentenceTransformer,
    code_embs: torch.Tensor,
    all_codes: List[str],
    items: List,
    get_text_fn,
    get_gold_fn,
    k: int,
    desc: str = "document-level",
) -> Dict[str, float]:
    recalls: List[float] = []
    precisions: List[float] = []
    mrrs: List[float] = []
    for item in tqdm(items, desc=desc, unit="doc"):
        gold = get_gold_fn(item)
        if not gold:
            continue
        text = get_text_fn(item)
        if not text:
            continue
        with torch.inference_mode():
            text_emb = model.encode(
                [text],
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=1,
            )
            scores = torch.matmul(text_emb, code_embs.T)[0]
            _, topk_idx = torch.topk(scores, k=min(k, scores.shape[0]))
        pred_codes = [all_codes[int(i)] for i in topk_idx.tolist()]
        recalls.append(recall_at_k(pred_codes, gold, k))
        precisions.append(precision_at_k(pred_codes, gold, k))
        mrrs.append(mrr_at_k(pred_codes, gold, k))
    n = len(recalls)
    if n == 0:
        return {f"R@{k}": 0.0, f"P@{k}": 0.0, f"MRR@{k}": 0.0, "n": 0}
    return {
        f"R@{k}": sum(recalls) / n,
        f"P@{k}": sum(precisions) / n,
        f"MRR@{k}": sum(mrrs) / n,
        "n": n,
    }


def evaluate_dataset(
    name: str,
    items: List,
    get_text_fn,
    get_gold_fn,
    annotator: EmbeddingAnnotator,
    competency_id_to_code: Dict[str, str],
    threshold: float,
    top_k: int,
    max_segment_length_for_context: int,
    rerank_top_k: int,
    confidence_aggregation: str,
    filter_segments: bool,
    k: int,
) -> Dict[str, float]:
    recalls: List[float] = []
    precisions: List[float] = []
    mrrs: List[float] = []
    skipped = 0
    for item in tqdm(items, desc=name, unit="doc"):
        gold = get_gold_fn(item)
        if not gold:
            skipped += 1
            continue
        text = get_text_fn(item)
        if not text:
            skipped += 1
            continue
        anns = annotator.annotate(
            text=text,
            threshold=threshold,
            top_k=top_k,
            max_segment_length_for_context=max_segment_length_for_context,
            rerank_top_k=rerank_top_k,
            confidence_aggregation=confidence_aggregation,
            filter_segments=filter_segments,
            use_cross_encoder_doc_score=False,
        )
        pred = _annotations_to_codes(anns, competency_id_to_code, max_codes=k)
        recalls.append(recall_at_k(pred, gold, k))
        precisions.append(precision_at_k(pred, gold, k))
        mrrs.append(mrr_at_k(pred, gold, k))
    n = len(recalls)
    if n == 0:
        return {f"R@{k}": 0.0, f"P@{k}": 0.0, f"MRR@{k}": 0.0, "n": 0, "skipped": skipped}
    return {
        f"R@{k}": sum(recalls) / n,
        f"P@{k}": sum(precisions) / n,
        f"MRR@{k}": sum(mrrs) / n,
        "n": n,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="R@20 по полному пайплайну и document-level")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Би-энкодер: путь к папке или HF id (например deepvk/USER-bge-m3). По умолчанию — первая папка из best/.",
    )
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_GISNAUKA)
    parser.add_argument(
        "--test-docs-csv",
        type=Path,
        default=DEFAULT_TEST_GISNAUKA_DOCS,
        help="Новый document-level test из VALID (gisnauka_samples_test_docs.csv).",
    )
    parser.add_argument("--gold-jsonl", type=Path, default=DEFAULT_GOLD_JSONL)
    parser.add_argument("--valid-csv", type=Path, default=DEFAULT_VALID_CSV)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-segment-context", type=int, default=DEFAULT_MAX_SEGMENT_LENGTH_FOR_CONTEXT)
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--confidence-aggregation", type=str, default=DEFAULT_CONFIDENCE_AGGREGATION)
    parser.add_argument("--no-filter-segments", action="store_true")
    parser.add_argument("--emb", type=Path, default=None)
    parser.add_argument("--k", type=int, default=EVAL_K)
    args = parser.parse_args()

    model_path = args.model if args.model is not None else _resolve_default_model()
    print(f"Би-энкодер: {model_path}")

    ontology_path = args.ontology
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology not found: {ontology_path}")

    competency_id_to_code = load_ontology_code_map(ontology_path)
    emb_path = args.emb
    if emb_path is not None and not emb_path.exists():
        emb_path = None

    annotator = EmbeddingAnnotator(
        ontology_path=ontology_path,
        model_name=model_path,
        cross_encoder_model=None,
        precomputed_embeddings_path=emb_path,
    )

    code_to_text = load_ontology_texts(ontology_path)
    all_codes = [c for c in code_to_text.keys() if is_leaf_grnti_code(c)]
    with torch.inference_mode():
        code_embs = annotator.model.encode(
            [code_to_text[c] for c in all_codes],
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=64,
        )

    threshold = args.threshold
    top_k = args.top_k
    max_seg = args.max_segment_context
    rerank_k = args.rerank_top_k
    conf_agg = args.confidence_aggregation
    filter_seg = not args.no_filter_segments
    k = args.k

    results: Dict[str, dict] = {}

    if args.test_csv.exists():
        items_csv = read_gold_csv(args.test_csv)
        if items_csv:
            res = evaluate_dataset(
                name="Test gisnauka (полный пайплайн)",
                items=items_csv,
                get_text_fn=lambda it: it.text or "",
                get_gold_fn=lambda it: list(it.gold_codes),
                annotator=annotator,
                competency_id_to_code=competency_id_to_code,
                threshold=threshold,
                top_k=top_k,
                max_segment_length_for_context=max_seg,
                rerank_top_k=rerank_k,
                confidence_aggregation=conf_agg,
                filter_segments=filter_seg,
                k=k,
            )
            results["test_gisnauka"] = res
            print(f"\n--- Test gisnauka (полный пайплайн): n={res['n']}, skipped={res.get('skipped', 0)} ---")
            print(f"  R@{k}: {res[f'R@{k}']:.4f}  P@{k}: {res[f'P@{k}']:.4f}  MRR@{k}: {res[f'MRR@{k}']:.4f}")
        if items_csv:
            doc_level = evaluate_doc_level(
                annotator.model,
                code_embs,
                all_codes,
                items_csv,
                get_text_fn=lambda it: it.text or "",
                get_gold_fn=lambda it: list(it.gold_codes),
                k=k,
                desc="Test gisnauka (document-level)",
            )
            results["test_gisnauka_doc_level"] = doc_level
            print(f"--- Test gisnauka (document-level): n={doc_level['n']} ---")
            print(f"  R@{k}: {doc_level[f'R@{k}']:.4f}  P@{k}: {doc_level[f'P@{k}']:.4f}  MRR@{k}: {doc_level[f'MRR@{k}']:.4f}")
    else:
        print(f"Test CSV not found: {args.test_csv}, skip.")

    # Новый document-level test, полученный из VALID
    if args.test_docs_csv.exists():
        items_docs = read_gold_csv(args.test_docs_csv)
        if items_docs:
            res_docs = evaluate_dataset(
                name="Test gisnauka docs (полный пайплайн)",
                items=items_docs,
                get_text_fn=lambda it: it.text or "",
                get_gold_fn=lambda it: list(it.gold_codes),
                annotator=annotator,
                competency_id_to_code=competency_id_to_code,
                threshold=threshold,
                top_k=top_k,
                max_segment_length_for_context=max_seg,
                rerank_top_k=rerank_k,
                confidence_aggregation=conf_agg,
                filter_segments=filter_seg,
                k=k,
            )
            results["test_gisnauka_docs"] = res_docs
            print(f"\n--- Test gisnauka docs (полный пайплайн): n={res_docs['n']}, skipped={res_docs.get('skipped', 0)} ---")
            print(f"  R@{k}: {res_docs[f'R@{k}']:.4f}  P@{k}: {res_docs[f'P@{k}']:.4f}  MRR@{k}: {res_docs[f'MRR@{k}']:.4f}")

        if items_docs:
            doc_level_docs = evaluate_doc_level(
                annotator.model,
                code_embs,
                all_codes,
                items_docs,
                get_text_fn=lambda it: it.text or "",
                get_gold_fn=lambda it: list(it.gold_codes),
                k=k,
                desc="Test gisnauka docs (document-level)",
            )
            results["test_gisnauka_docs_doc_level"] = doc_level_docs
            print(f"--- Test gisnauka docs (document-level): n={doc_level_docs['n']} ---")
            print(f"  R@{k}: {doc_level_docs[f'R@{k}']:.4f}  P@{k}: {doc_level_docs[f'P@{k}']:.4f}  MRR@{k}: {doc_level_docs[f'MRR@{k}']:.4f}")
    else:
        print(f"Test docs CSV not found: {args.test_docs_csv}, skip.")

    if args.gold_jsonl.exists():
        items_jsonl = read_gold_jsonl(args.gold_jsonl)
        if items_jsonl:

            def get_text_jsonl(it: GoldItem) -> str:
                if it.text:
                    return it.text
                if it.text_path and it.text_path.exists():
                    return it.text_path.read_text(encoding="utf-8", errors="replace")
                return ""

            res = evaluate_dataset(
                name="Gold JSONL (полный пайплайн)",
                items=items_jsonl,
                get_text_fn=get_text_jsonl,
                get_gold_fn=lambda it: list(it.gold_codes),
                annotator=annotator,
                competency_id_to_code=competency_id_to_code,
                threshold=threshold,
                top_k=top_k,
                max_segment_length_for_context=max_seg,
                rerank_top_k=rerank_k,
                confidence_aggregation=conf_agg,
                filter_segments=filter_seg,
                k=k,
            )
            results["gold_jsonl"] = res
            print(f"\n--- Gold JSONL (полный пайплайн): n={res['n']}, skipped={res.get('skipped', 0)} ---")
            print(f"  R@{k}: {res[f'R@{k}']:.4f}  P@{k}: {res[f'P@{k}']:.4f}  MRR@{k}: {res[f'MRR@{k}']:.4f}")
        doc_level_j = evaluate_doc_level(
            annotator.model,
            code_embs,
            all_codes,
            items_jsonl,
            get_text_fn=get_text_jsonl,
            get_gold_fn=lambda it: list(it.gold_codes),
            k=k,
            desc="Gold JSONL (document-level)",
        )
        results["gold_jsonl_doc_level"] = doc_level_j
        print(f"--- Gold JSONL (document-level): n={doc_level_j['n']} ---")
        print(f"  R@{k}: {doc_level_j[f'R@{k}']:.4f}  P@{k}: {doc_level_j[f'P@{k}']:.4f}  MRR@{k}: {doc_level_j[f'MRR@{k}']:.4f}")
    else:
        print(f"Gold JSONL not found: {args.gold_jsonl}, skip.")

    if args.valid_csv.exists():
        items_valid = read_gold_csv(args.valid_csv)
        if items_valid:
            res_v = evaluate_dataset(
                name="Valid (полный пайплайн)",
                items=items_valid,
                get_text_fn=lambda it: it.text or "",
                get_gold_fn=lambda it: list(it.gold_codes),
                annotator=annotator,
                competency_id_to_code=competency_id_to_code,
                threshold=threshold,
                top_k=top_k,
                max_segment_length_for_context=max_seg,
                rerank_top_k=rerank_k,
                confidence_aggregation=conf_agg,
                filter_segments=filter_seg,
                k=k,
            )
            results["valid"] = res_v
            print(f"\n--- Valid (полный пайплайн): n={res_v['n']}, skipped={res_v.get('skipped', 0)} ---")
            print(f"  R@{k}: {res_v[f'R@{k}']:.4f}  P@{k}: {res_v[f'P@{k}']:.4f}  MRR@{k}: {res_v[f'MRR@{k}']:.4f}")
        if items_valid:
            doc_level_v = evaluate_doc_level(
                annotator.model,
                code_embs,
                all_codes,
                items_valid,
                get_text_fn=lambda it: it.text or "",
                get_gold_fn=lambda it: list(it.gold_codes),
                k=k,
                desc="Valid (document-level)",
            )
            results["valid_doc_level"] = doc_level_v
            print(f"--- Valid (document-level): n={doc_level_v['n']} ---")
            print(f"  R@{k}: {doc_level_v[f'R@{k}']:.4f}  P@{k}: {doc_level_v[f'P@{k}']:.4f}  MRR@{k}: {doc_level_v[f'MRR@{k}']:.4f}")
    else:
        print(f"Valid CSV not found: {args.valid_csv}, skip.")

    print("\n--- Summary ---")
    for key, res in results.items():
        print(f"  {key}: R@{k}={res.get(f'R@{k}', 0):.4f}  P@{k}={res.get(f'P@{k}', 0):.4f}  MRR@{k}={res.get(f'MRR@{k}', 0):.4f}  n={res.get('n', 0)}")


if __name__ == "__main__":
    main()
