from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from datasets import Dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TRAIN_SEGMENTS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_train_augmented.csv"
VALID_SEGMENTS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_valid_augmented.csv"
TEST_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test.csv"
ONTOLOGY_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
OUTPUT_DIR = PROJECT_DIR / "models" / "bi-encoder-gisnauka-trainer"
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


def evaluate_on_test(model_ref: str) -> Dict[str, float]:
    import csv

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


def build_pairs_from_segments(
    segments_csv: Path,
    code_to_text: Dict[str, str],
) -> List[InputExample]:
    """
    Строит обучающие пары (segment_text, code_text) из предварительно
    сегментированного CSV для всех doc_id в TRAIN_SEGMENTS_CSV.
    """
    import csv

    if not segments_csv.exists():
        raise FileNotFoundError(f"segments CSV not found: {segments_csv}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    examples: List[InputExample] = []

    with segments_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Если задано, ограничивает число обучающих пар (после перемешивания).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SentenceTransformer(args.base_model, device=device)

    code_to_text = load_ontology_texts(ONTOLOGY_PATH)
    train_examples = build_pairs_from_segments(TRAIN_SEGMENTS_CSV, code_to_text)
    valid_examples = build_pairs_from_segments(VALID_SEGMENTS_CSV, code_to_text)
    if not train_examples:
        raise RuntimeError("no training pairs built")
    if not valid_examples:
        raise RuntimeError("no validation pairs built")

    print(f"Train pairs (из сегментов): {len(train_examples)}")
    print(f"Valid pairs (из сегментов): {len(valid_examples)}")

    # Преобразуем список InputExample в HF Dataset с двумя текстовыми колонками
    train_dataset = Dataset.from_list(
        [{"text1": ex.texts[0], "text2": ex.texts[1]} for ex in train_examples]
    )
    valid_dataset = Dataset.from_list(
        [{"text1": ex.texts[0], "text2": ex.texts[1]} for ex in valid_examples]
    )

    if args.max_train_samples is not None:
        max_n = int(args.max_train_samples)
        if max_n <= 0:
            raise ValueError("--max-train-samples must be positive")
        if len(train_dataset) > max_n:
            train_dataset = train_dataset.shuffle(seed=int(args.seed)).select(range(max_n))
        print(f"Train dataset ограничен до: {len(train_dataset)}")

    train_loss = losses.MultipleNegativesRankingLoss(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(args.epochs),
        per_device_train_batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        warmup_ratio=float(args.warmup_ratio),
        seed=int(args.seed),
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=2,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        loss=train_loss,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"Model saved to: {output_dir}")

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
        "params": {
            "base_model": args.base_model,
            "output_dir": str(output_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "max_train_samples": args.max_train_samples,
            "seed": args.seed,
        },
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved run result to: {result_path}")


if __name__ == "__main__":
    main()

