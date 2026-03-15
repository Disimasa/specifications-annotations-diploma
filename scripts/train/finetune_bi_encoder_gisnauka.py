from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses

# первый раз
# python scripts/train/finetune_bi_encoder_gisnauka.py --output-dir models/bi-encoder-gisnauka-200k --max-steps 40000
# последующие разы
# python scripts/train/finetune_bi_encoder_gisnauka.py --resume models/bi-encoder-gisnauka-200k --output-dir models/bi-encoder-gisnauka-200k --max-steps 40000


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TRAIN_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_train_augmented_clean.csv"
TRAIN_SEGMENTS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_train_augmented.csv"
VALID_SEGMENTS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_valid_augmented.csv"
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
    code_to_text: Dict[str, str],
) -> Tuple[List[InputExample], List[str]]:
    """
    Строит обучающие пары (segment_text, code_text) из предварительно
    сегментированного CSV для всех doc_id в TRAIN_SEGMENTS_CSV.
    """
    import csv
    import sys

    if not TRAIN_SEGMENTS_CSV.exists():
        raise FileNotFoundError(f"segments CSV not found: {TRAIN_SEGMENTS_CSV}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    examples: List[InputExample] = []
    example_doc_ids: List[str] = []

    with TRAIN_SEGMENTS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            doc_id = (row.get("doc_id") or "").strip() or f"train_{i}"
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
                example_doc_ids.append(doc_id)
    return examples, example_doc_ids


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


def evaluate_encoder_on_segmented_csv(
    model: SentenceTransformer,
    segments_csv: Path,
    code_to_text: Dict[str, str],
    *,
    k: int = 20,
    segment_batch_size: int = 64,
) -> Dict[str, float]:
    import csv
    import sys

    if not segments_csv.exists():
        raise FileNotFoundError(f"segments CSV not found: {segments_csv}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

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

    docs_segments: Dict[str, List[str]] = {}
    docs_gold: Dict[str, List[str]] = {}

    with segments_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            doc_id = (row.get("doc_id") or "").strip() or f"valid_{i}"
            seg = (row.get("segment_text") or "").strip()
            if not seg:
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
            docs_segments.setdefault(doc_id, []).append(seg)
            if doc_id not in docs_gold:
                docs_gold[doc_id] = gold_codes

    recalls: List[float] = []
    precisions: List[float] = []

    for doc_id, segments in docs_segments.items():
        gold_codes = docs_gold.get(doc_id, [])
        if not gold_codes:
            continue
        with torch.inference_mode():
            seg_embs = model.encode(
                segments,
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=int(segment_batch_size),
            )
            scores = torch.matmul(seg_embs, code_embs.T)
            doc_scores = scores.max(dim=0).values
            topk_idx = torch.topk(doc_scores, k=min(int(k), int(doc_scores.shape[0]))).indices
        pred_codes = [all_codes[int(i)] for i in topk_idx.tolist()]
        recalls.append(recall_at_k(pred_codes, gold_codes, int(k)))
        precisions.append(precision_at_k(pred_codes, gold_codes, int(k)))

    macro_r20 = sum(recalls) / float(len(recalls)) if recalls else 0.0
    macro_p20 = sum(precisions) / float(len(precisions)) if precisions else 0.0
    return {"R@20": macro_r20, "P@20": macro_p20}


class UniquePairsBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Батчер по индексам `InputExample`, который гарантирует, что
    в рамках одного батча не повторяются:
    - тексты сегментов (texts[0])
    - тексты специализаций (texts[1])
    """

    def __init__(
        self,
        data: Sequence[InputExample],
        doc_ids: Sequence[str],
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.data = data
        self.doc_ids = doc_ids
        self.batch_size = int(batch_size)
        self.generator = generator

    def __iter__(self):
        n = len(self.data)
        if n == 0:
            return

        if self.generator is None:
            indices = torch.randperm(n).tolist()
        else:
            indices = torch.randperm(n, generator=self.generator).tolist()

        i = 0
        while i < n:
            used_segments = set()
            used_codes = set()
            used_docs = set()
            batch: List[int] = []

            while i < n and len(batch) < self.batch_size:
                idx = indices[i]
                i += 1
                ex = self.data[idx]
                doc_id = self.doc_ids[idx] if idx < len(self.doc_ids) else None
                if not ex.texts or len(ex.texts) < 2:
                    continue
                seg_text = ex.texts[0]
                code_text = ex.texts[1]
                if (
                    seg_text in used_segments
                    or code_text in used_codes
                    or (doc_id is not None and doc_id in used_docs)
                ):
                    continue
                used_segments.add(seg_text)
                used_codes.add(code_text)
                if doc_id is not None:
                    used_docs.add(doc_id)
                batch.append(idx)

            if batch:
                yield batch

    def __len__(self) -> int:
        if not self.data or self.batch_size <= 0:
            return 0
        return (len(self.data) + self.batch_size - 1) // self.batch_size


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to continue training from (overrides --base-model)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="If set, train at most this many steps then save and exit (for multi-session runs)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="If set, train at most this many samples per run then save and exit (for multi-session runs; use with --resume to cover all data)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    steps_done = 0
    samples_done = 0
    if args.resume.strip():
        resume_path = Path(args.resume.strip())
        model = SentenceTransformer(str(resume_path), device=device)
        print(f"Resumed from: {args.resume}")
        state_path = resume_path / "training_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            steps_done = int(state.get("steps_done", 0))
            samples_done = int(state.get("samples_done", 0))
            print(f"Resuming from step {steps_done}, samples {samples_done} (next run will continue with the following data)")
    else:
        model = SentenceTransformer(args.base_model, device=device)

    code_to_text = load_ontology_texts(ONTOLOGY_PATH)
    train_examples, train_doc_ids = build_train_pairs_from_segments(code_to_text)
    if not train_examples:
        raise RuntimeError("no training pairs built")

    print(f"Train pairs (из сегментов): {len(train_examples)}")

    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Кастомный батчер: в пределах батча уникальные сегменты, специализации и документы
    g = torch.Generator()
    g.manual_seed(int(args.seed))
    batch_sampler = UniquePairsBatchSampler(
        train_examples,
        doc_ids=train_doc_ids,
        batch_size=int(args.batch_size),
        generator=g,
    )

    class _BatchIterator:
        def __init__(self, data, sampler):
            self.data = data
            self.sampler = sampler

        def __iter__(self):
            for batch_indices in self.sampler:
                yield [self.data[i] for i in batch_indices]

        def __len__(self):
            return len(self.sampler)

    train_dataloader = _BatchIterator(train_examples, batch_sampler)

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * int(args.epochs)
    max_steps_this_run = None
    max_train_samples_this_run = None
    if args.max_train_samples is not None and args.max_train_samples > 0:
        max_train_samples_this_run = int(args.max_train_samples)
        # Оценка шагов для шедулера (батчи разного размера)
        est_steps_this_run = (max_train_samples_this_run + int(args.batch_size) - 1) // int(args.batch_size)
        total_steps = min(total_steps, steps_done + est_steps_this_run)
    if args.max_steps is not None and args.max_steps > 0:
        max_steps_this_run = int(args.max_steps)
        total_steps = min(total_steps, steps_done + max_steps_this_run)
    warmup_steps = max(10, int(0.1 * (total_steps or 1)))
    print(f"Steps per epoch: {steps_per_epoch}, steps_done: {steps_done}, samples_done: {samples_done}, total steps (this run): {total_steps}, warmup steps: {warmup_steps}")
    if max_train_samples_this_run is not None:
        print(f"Max train samples this run: {max_train_samples_this_run}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_r20 = -1.0
    best_epoch = 0
    bad_epochs = 0

    # Оптимизатор и шедулер
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = steps_done
    samples_this_run = 0
    stop_requested = False
    for _ in range(steps_done):
        scheduler.step()

    def _skip_batches(dl, n: int):
        it = iter(dl)
        skipped = 0
        while skipped < n:
            try:
                next(it)
                skipped += 1
            except StopIteration:
                it = iter(dl)
        return it

    def _skip_until_samples(dl, target: int):
        it = iter(dl)
        total_skipped = 0
        while total_skipped < target:
            try:
                batch = next(it)
                total_skipped += len(batch)
            except StopIteration:
                it = iter(dl)
        return it

    if max_train_samples_this_run is not None and samples_done > 0:
        train_iter = _skip_until_samples(train_dataloader, samples_done)
    elif steps_done > 0:
        train_iter = _skip_batches(train_dataloader, steps_done)
    else:
        train_iter = iter(train_dataloader)
    epoch = 1
    while epoch <= int(args.epochs) and not stop_requested:
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        while True:
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            sentence_features, labels = model.smart_batching_collate(batch)
            for sent_feat in sentence_features:
                for k, v in list(sent_feat.items()):
                    if isinstance(v, torch.Tensor):
                        sent_feat[k] = v.to(device)

            loss_value = train_loss(sentence_features, labels)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += float(loss_value.detach().cpu())
            epoch_steps += 1
            global_step += 1
            batch_samples = len(batch)
            samples_this_run += batch_samples
            if max_train_samples_this_run is not None and samples_this_run >= max_train_samples_this_run:
                stop_requested = True
                break
            if max_steps_this_run is not None and global_step >= steps_done + max_steps_this_run:
                stop_requested = True
                break

        print(f"Epoch {epoch} train loss: {epoch_loss / max(1, epoch_steps):.4f}")

        valid_metrics = evaluate_encoder_on_segmented_csv(
            model,
            VALID_SEGMENTS_CSV,
            code_to_text,
            k=20,
            segment_batch_size=64,
        )
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
        if stop_requested:
            if max_train_samples_this_run is not None:
                print(f"Reached max_train_samples={max_train_samples_this_run} (saw {samples_this_run} samples). Saving and exiting.")
            else:
                print(f"Reached max_steps={max_steps_this_run}. Saving and exiting.")
            break
        epoch += 1
        train_iter = iter(train_dataloader)

    model.save(str(output_dir))
    final_samples_done = samples_done + samples_this_run
    (output_dir / "training_state.json").write_text(
        json.dumps({"steps_done": global_step, "samples_done": final_samples_done}, indent=2),
        encoding="utf-8",
    )
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
        "valid_best": {"epoch": best_epoch, "R@20": best_r20},
        "params": {
            "base_model": args.base_model,
            "output_dir": str(output_dir),
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "max_train_samples": args.max_train_samples,
            "steps_done": global_step,
            "samples_done": final_samples_done,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "early_stop_patience": args.early_stop_patience,
            "seed": args.seed,
        },
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved run result to: {result_path}")


if __name__ == "__main__":
    main()



