from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TRAIN_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_train.csv"
TRAIN_SEGMENTS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_train_filtered.csv"
TEST_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test.csv"
ONTOLOGY_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
OUTPUT_DIR = PROJECT_DIR / "models" / "bi-encoder-gisnauka"
RUNS_DIR = PROJECT_DIR / "data" / "gold" / "bi_encoder_runs"

BASE_MODEL = "deepvk/USER-bge-m3"


def _is_leaf_grnti_code(code: str) -> bool:
    parts = code.split(".")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


def load_ontology_texts(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    code_to_text: Dict[str, str] = {}
    for n in nodes:
        code = (n.get("code") or "").strip()
        if not code:
            continue
        full_label = (n.get("full_label") or n.get("label") or "").strip()
        llm_desc = (n.get("llm_description") or "").strip()
        parts: List[str] = []
        if full_label:
            parts.append(full_label)
        if llm_desc:
            parts.append(llm_desc)
        text = ". ".join(parts).strip(" .")
        if not text:
            text = full_label or code
        code_to_text[code] = text
    return code_to_text


def load_train_examples() -> List[InputExample]:
    """
    Устаревшая функция (обучение по полным текстам), оставлена только на случай отладки.
    Сейчас тренер использует предварительно сегментированный CSV.
    """
    import csv
    import sys

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"train CSV not found: {TRAIN_CSV}")
    if not ONTOLOGY_PATH.exists():
        raise FileNotFoundError(f"ontology not found: {ONTOLOGY_PATH}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    code_to_text = load_ontology_texts(ONTOLOGY_PATH)

    examples: List[InputExample] = []

    with TRAIN_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            text = f"{title}\n\n{abstract}".strip()
            if not text:
                continue
            codes_raw = (row.get("grnti_codes") or "").strip()
            if not codes_raw:
                continue
            codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
            codes = [c for c in codes if _is_leaf_grnti_code(c)]
            if not codes:
                continue

            for code in codes:
                comp_text = code_to_text.get(code)
                if not comp_text:
                    continue
                examples.append(InputExample(texts=[text, comp_text]))

    return examples


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


def evaluate_on_test(model_ref: str) -> Dict[str, float]:
    import csv
    import sys

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"test CSV not found: {TEST_CSV}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    code_to_text = load_ontology_texts(ONTOLOGY_PATH)
    all_codes: List[str] = [c for c in code_to_text.keys() if _is_leaf_grnti_code(c)]
    if not all_codes:
        raise RuntimeError("no leaf codes found in ontology")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidate_path = Path(model_ref)
    if candidate_path.exists():
        model_name = str(candidate_path)
    else:
        model_name = model_ref
    model = SentenceTransformer(model_name, device=device)

    code_texts = [code_to_text[c] for c in all_codes]
    with torch.inference_mode():
        code_embs = model.encode(
            code_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=64,
        )

    eval_k = 20
    recalls: List[float] = []
    precisions: List[float] = []

    with TEST_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            text = f"{title}\n\n{abstract}".strip()
            if not text:
                continue
            codes_raw = (row.get("grnti_codes") or "").strip()
            if not codes_raw:
                continue
            gold_codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
            gold_codes = [c for c in gold_codes if _is_leaf_grnti_code(c)]
            gold_codes = sorted(set(gold_codes))
            if not gold_codes:
                continue
            with torch.inference_mode():
                text_emb = model.encode(
                    [text],
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    batch_size=1,
                )
                scores = torch.matmul(text_emb, code_embs.T)[0]
                topk_vals, topk_idx = torch.topk(scores, k=min(eval_k, scores.shape[0]))
            pred_codes = [all_codes[int(i)] for i in topk_idx.tolist()]

            r20 = recall_at_k(pred_codes, gold_codes, eval_k)
            p20 = precision_at_k(pred_codes, gold_codes, eval_k)
            recalls.append(r20)
            precisions.append(p20)

    macro_r20 = sum(recalls) / float(len(recalls)) if recalls else 0.0
    macro_p20 = sum(precisions) / float(len(precisions)) if precisions else 0.0
    return {"R@20": macro_r20, "P@20": macro_p20}


def _load_docs_from_csv(
    csv_path: Path,
    code_to_text: Dict[str, str],
    *,
    max_docs: int | None,
    seed: int,
) -> List[Tuple[str, str, List[str]]]:
    import csv
    import random

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    docs: List[Tuple[str, str, List[str]]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            text = f"{title}\n\n{abstract}".strip()
            if not text:
                continue
            codes_raw = (row.get("grnti_codes") or "").strip()
            if not codes_raw:
                continue
            gold_codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
            gold_codes = [c for c in gold_codes if _is_leaf_grnti_code(c)]
            gold_codes = [c for c in gold_codes if c in code_to_text]
            gold_codes = sorted(set(gold_codes))
            if not gold_codes:
                continue
            doc_id = (row.get("doc_id") or "").strip() or f"gisnauka_{i}"
            docs.append((doc_id, text, gold_codes))

    rnd = random.Random(int(seed))
    rnd.shuffle(docs)
    if max_docs is not None:
        docs = docs[: int(max_docs)]
    return docs


def _split_train_valid_docs(
    docs: List[Tuple[str, str, List[str]]],
    *,
    valid_docs: int,
) -> Tuple[List[Tuple[str, str, List[str]]], List[Tuple[str, str, List[str]]]]:
    if valid_docs <= 0:
        return docs, []
    if valid_docs >= len(docs):
        valid_docs = max(1, len(docs) // 5)
    valid = docs[:valid_docs]
    train = docs[valid_docs:]
    return train, valid


def build_train_pairs_from_docs(
    docs: List[Tuple[str, str, List[str]]],
    code_to_text: Dict[str, str],
) -> List[InputExample]:
    examples: List[InputExample] = []
    for _doc_id, text, gold_codes in docs:
        for code in gold_codes:
            comp_text = code_to_text.get(code)
            if not comp_text:
                continue
            examples.append(InputExample(texts=[text, comp_text]))
    return examples


def build_train_pairs_from_segments(
    docs: List[Tuple[str, str, List[str]]],
    code_to_text: Dict[str, str],
) -> List[InputExample]:
    """
    Строит обучающие пары (segment_text, code_text) из предварительно
    сегментированного CSV, используя только те doc_id, которые попали в train_docs.
    """
    import csv
    import sys

    if not TRAIN_SEGMENTS_CSV.exists():
        raise FileNotFoundError(f"segments CSV not found: {TRAIN_SEGMENTS_CSV}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    train_doc_ids = {doc_id for doc_id, _text, _codes in docs}
    examples: List[InputExample] = []

    with TRAIN_SEGMENTS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = (row.get("doc_id") or "").strip()
            if doc_id not in train_doc_ids:
                continue
            segment_text = (row.get("segment_text") or "").strip()
            if not segment_text:
                continue
            codes_raw = (row.get("grnti_codes") or "").strip()
            if not codes_raw:
                continue
            codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
            codes = [c for c in codes if _is_leaf_grnti_code(c)]
            if not codes:
                continue
            for code in codes:
                comp_text = code_to_text.get(code)
                if not comp_text:
                    continue
                examples.append(InputExample(texts=[segment_text, comp_text]))
    return examples


def evaluate_encoder_on_docs(
    model: SentenceTransformer,
    docs: List[Tuple[str, str, List[str]]],
    code_to_text: Dict[str, str],
    *,
    k: int = 20,
) -> Dict[str, float]:
    all_codes: List[str] = [c for c in code_to_text.keys() if _is_leaf_grnti_code(c)]
    if not all_codes:
        return {"R@20": 0.0, "P@20": 0.0}

    code_texts = [code_to_text[c] for c in all_codes]
    with torch.inference_mode():
        code_embs = model.encode(
            code_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=64,
        )

    recalls: List[float] = []
    precisions: List[float] = []

    for _doc_id, text, gold_codes in docs:
        with torch.inference_mode():
            text_emb = model.encode(
                [text],
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=1,
            )
            scores = torch.matmul(text_emb, code_embs.T)[0]
            topk_idx = torch.topk(scores, k=min(int(k), int(scores.shape[0]))).indices
        pred_codes = [all_codes[int(i)] for i in topk_idx.tolist()]
        recalls.append(recall_at_k(pred_codes, gold_codes, int(k)))
        precisions.append(precision_at_k(pred_codes, gold_codes, int(k)))

    macro_r20 = sum(recalls) / float(len(recalls)) if recalls else 0.0
    macro_p20 = sum(precisions) / float(len(precisions)) if precisions else 0.0
    return {"R@20": macro_r20, "P@20": macro_p20}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-train-docs", type=int, default=50)
    parser.add_argument("--valid-docs", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SentenceTransformer(args.base_model, device=device)

    code_to_text = load_ontology_texts(ONTOLOGY_PATH)
    docs_all = _load_docs_from_csv(
        TRAIN_CSV,
        code_to_text,
        max_docs=int(args.max_train_docs) if int(args.max_train_docs) > 0 else None,
        seed=int(args.seed),
    )
    if not docs_all:
        raise RuntimeError("no train docs loaded")
    train_docs, valid_docs = _split_train_valid_docs(docs_all, valid_docs=int(args.valid_docs))
    train_examples = build_train_pairs_from_segments(train_docs, code_to_text)
    if not train_examples:
        raise RuntimeError("no training pairs built")

    print(f"Train docs: {len(train_docs)}")
    print(f"Valid docs: {len(valid_docs)}")
    print(f"Train pairs: {len(train_examples)}")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    total_steps = len(train_dataloader) * int(args.epochs)
    warmup_steps = max(10, int(0.1 * total_steps))
    print(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_r20 = -1.0
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps if epoch == 1 else 0,
            optimizer_params={"lr": args.learning_rate},
            use_amp=torch.cuda.is_available(),
            show_progress_bar=True,
        )

        if valid_docs:
            valid_metrics = evaluate_encoder_on_docs(model, valid_docs, code_to_text, k=20)
            r20 = float(valid_metrics.get("R@20", 0.0))
            print(f"Valid R@20 after epoch {epoch}: {r20:.4f}")
            if r20 > best_r20 + 1e-9:
                best_r20 = r20
                best_epoch = epoch
                bad_epochs = 0
                model.save(str(best_dir))
            else:
                bad_epochs += 1
                if bad_epochs > int(args.early_stop_patience):
                    print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch} R@20={best_r20:.4f}")
                    break

    model.save(str(output_dir))
    print(f"Model saved to: {output_dir}")
    if best_dir.exists():
        print(f"Best checkpoint: {best_dir}")

    original_metrics = evaluate_on_test(args.base_model)
    finetuned_metrics = evaluate_on_test(str(output_dir))
    print("Original encoder:")
    print(f"  R@20: {original_metrics.get('R@20', 0.0):.4f}")
    print(f"  P@20: {original_metrics.get('P@20', 0.0):.4f}")
    print("Finetuned encoder:")
    print(f"  R@20: {finetuned_metrics.get('R@20', 0.0):.4f}")
    print(f"  P@20: {finetuned_metrics.get('P@20', 0.0):.4f}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    payload = {
        "original_metrics": original_metrics,
        "finetuned_metrics": finetuned_metrics,
        "valid_best": {"epoch": best_epoch, "R@20": best_r20} if valid_docs else None,
        "params": {
            "base_model": args.base_model,
            "output_dir": str(output_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_train_docs": args.max_train_docs,
            "valid_docs": args.valid_docs,
            "early_stop_patience": args.early_stop_patience,
            "seed": args.seed,
        },
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved run result to: {result_path}")


if __name__ == "__main__":
    main()



