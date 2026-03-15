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
    """R@20 и P@20 на тесте через пайплайн аннотирования (как в scripts/eval/evaluate_r20_full_pipeline.py)."""
    eval_dir = PROJECT_DIR / "scripts" / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from evaluate_r20_full_pipeline import evaluate_dataset  # noqa: E402

    from annotation.annotator import EmbeddingAnnotator
    from lib.eval_defaults import (
        DEFAULT_CONFIDENCE_AGGREGATION,
        DEFAULT_FILTER_SEGMENTS,
        DEFAULT_MAX_SEGMENT_LENGTH_FOR_CONTEXT,
        DEFAULT_RERANK_TOP_K,
        DEFAULT_THRESHOLD,
        DEFAULT_TOP_K,
        EVAL_K,
    )
    from lib.gold_io import read_gold_csv
    from lib.grnti_ontology import load_ontology_code_map

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"test CSV not found: {TEST_CSV}")
    if not ONTOLOGY_PATH.exists():
        raise FileNotFoundError(f"ontology not found: {ONTOLOGY_PATH}")

    competency_id_to_code = load_ontology_code_map(ONTOLOGY_PATH)
    annotator = EmbeddingAnnotator(
        ontology_path=ONTOLOGY_PATH,
        model_name=model_ref,
        cross_encoder_model=None,
        precomputed_embeddings_path=None,
    )
    items = read_gold_csv(TEST_CSV)
    if not items:
        return {"R@20": 0.0, "P@20": 0.0}

    k = EVAL_K
    res = evaluate_dataset(
        name="Test (пайплайн)",
        items=items,
        get_text_fn=lambda it: it.text or "",
        get_gold_fn=lambda it: list(it.gold_codes),
        annotator=annotator,
        competency_id_to_code=competency_id_to_code,
        threshold=DEFAULT_THRESHOLD,
        top_k=DEFAULT_TOP_K,
        max_segment_length_for_context=DEFAULT_MAX_SEGMENT_LENGTH_FOR_CONTEXT,
        rerank_top_k=DEFAULT_RERANK_TOP_K,
        confidence_aggregation=DEFAULT_CONFIDENCE_AGGREGATION,
        filter_segments=DEFAULT_FILTER_SEGMENTS,
        k=k,
    )
    return {
        "R@20": res[f"R@{k}"],
        "P@20": res[f"P@{k}"],
    }


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
    parser.add_argument("--resume", type=str, default="", help="Путь к чекпоинту для продолжения обучения (режим по сэмплам с --max-train-samples)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Макс. сэмплов за один запуск; с --resume следующий чанк без дублей (обучение на всех данных по частям).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    samples_done = 0
    if args.resume.strip():
        resume_path = Path(args.resume.strip())
        model = SentenceTransformer(str(resume_path), device=device)
        print(f"Resumed from: {args.resume}")
        state_path = resume_path / "training_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            samples_done = int(state.get("samples_done", 0))
            print(f"Resuming from samples_done={samples_done} (следующий чанк)")
    else:
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

    # Преобразуем в HF Dataset и детерминированно перемешиваем
    full_train_dataset = Dataset.from_list(
        [{"text1": ex.texts[0], "text2": ex.texts[1]} for ex in train_examples]
    )
    full_train_dataset = full_train_dataset.shuffle(seed=int(args.seed))
    valid_dataset = Dataset.from_list(
        [{"text1": ex.texts[0], "text2": ex.texts[1]} for ex in valid_examples]
    )

    N = len(full_train_dataset)
    max_n = int(args.max_train_samples) if args.max_train_samples is not None else None

    if max_n is not None and max_n <= 0:
        raise ValueError("--max-train-samples must be positive")

    if args.resume.strip():
        # Следующий чанк: [samples_done : samples_done + max_train_samples]
        if max_n is None:
            raise ValueError("при --resume укажите --max-train-samples")
        start = samples_done
        end = min(samples_done + max_n, N)
        if start >= N:
            print("Уже обучено на всех сэмплах. Выход.")
            return
        if start >= end:
            print("Нет новых сэмплов для этого чанка. Выход.")
            return
        train_dataset = full_train_dataset.select(range(start, end))
        print(f"Чанк для этого запуска: сэмплы {start}–{end} (всего {len(train_dataset)})")
    elif max_n is not None:
        # Первый запуск в режиме по чанкам: первые max_n сэмплов
        train_dataset = full_train_dataset.select(range(0, min(max_n, N)))
        print(f"Train dataset (первый чанк): {len(train_dataset)} сэмплов")
    else:
        train_dataset = full_train_dataset
        print(f"Train dataset: {len(train_dataset)} сэмплов")

    train_loss = losses.MultipleNegativesRankingLoss(model)

    base_output_dir = Path(args.output_dir)
    if args.resume.strip():
        output_dir = Path(args.resume.strip())
    elif max_n is not None:
        output_dir = base_output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir / timestamp
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
    final_samples_done = samples_done + len(train_dataset)
    (output_dir / "training_state.json").write_text(
        json.dumps({"samples_done": final_samples_done}, indent=2),
        encoding="utf-8",
    )
    print(f"Model saved to: {output_dir}, samples_done={final_samples_done}")

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
            "base_output_dir": str(base_output_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "max_train_samples": args.max_train_samples,
            "samples_done": final_samples_done,
            "seed": args.seed,
        },
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved run result to: {result_path}")


if __name__ == "__main__":
    main()

