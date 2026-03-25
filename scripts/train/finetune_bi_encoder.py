from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import torch
from datasets import Dataset
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.sampler import NoDuplicatesBatchSampler
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from transformers import TrainerCallback

# Колонки, без которых HierarchicalGrntiBatchSampler / precomputed factory не имеют смысла.
_HIERARCHICAL_SAMPLER_META_COLS = frozenset({"doc_id", "leaf", "parent", "grand", "doc_gold_leaves"})


class FinetuneBiEncoderTrainer(SentenceTransformerTrainer):
    """
    Кастомный callable batch_sampler (иерархический / предбатчи) в ST применяется и к eval/test
    через get_batch_sampler(eval_dataset). Валид — только пары text1/text2, без GRNTI-метаданных,
    поэтому без подмены падаем с «не хватает колонок».
    """

    def __init__(self, *args, debug_collator_meta_tokenization: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.debug_collator_meta_tokenization = bool(debug_collator_meta_tokenization)
        self._debug_collator_printed = False

    def get_batch_sampler(self, dataset, batch_size, drop_last, valid_label_columns=None, generator=None, seed=0):
        if callable(self.args.batch_sampler):
            # Важное: переопределение должно различать:
            # - HierarchicalGrntiBatchSampler (нужны meta-колонки)
            # - PrecomputedEpochBatchSampler (meta-колонки не нужны, но нужен match размера dataset)
            from lib.hierarchical_grnti_batch_sampler import HierarchicalGrntiBatchSamplerFactory
            from lib.precomputed_epoch_batch_sampler import PrecomputedEpochBatchSamplerFactory

            bs = self.args.batch_sampler
            if isinstance(bs, HierarchicalGrntiBatchSamplerFactory):
                cols = set(dataset.column_names)
                if not _HIERARCHICAL_SAMPLER_META_COLS.issubset(cols):
                    return NoDuplicatesBatchSampler(
                        dataset,
                        batch_size=batch_size,
                        drop_last=drop_last,
                        valid_label_columns=valid_label_columns,
                        generator=generator,
                        seed=seed,
                    )
            elif isinstance(bs, PrecomputedEpochBatchSamplerFactory):
                stored_size = getattr(bs, "stored_size", None)
                if stored_size is not None and len(dataset) != int(stored_size):
                    # Eval dataset (text1/text2) имеет другой размер, поэтому precomputed sampler использовать нельзя.
                    return NoDuplicatesBatchSampler(
                        dataset,
                        batch_size=batch_size,
                        drop_last=drop_last,
                        valid_label_columns=valid_label_columns,
                        generator=generator,
                        seed=seed,
                    )
        return super().get_batch_sampler(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        # Диагностика: SentenceTransformerDataCollator токенизирует ВСЕ string-колонки датасета
        # и SentenceTransformerTrainer отдаёт их в loss как sentence_features по ключам *_input_ids.
        # Если кроме (text1,text2) токенизируются meta-колонки (doc_id/leaf/parent/...), loss
        # начинает оптимизировать не ту задачу, что и baseline.
        if self.debug_collator_meta_tokenization and not self._debug_collator_printed:
            feature_prefixes = set()
            for k in inputs.keys():
                if k.endswith("_input_ids"):
                    feature_prefixes.add(k[: -len("_input_ids")])

            sorted_prefixes = sorted(feature_prefixes)
            print(f"[debug collator] tokenized feature prefixes: {sorted_prefixes}")

            meta_prefixes = sorted(feature_prefixes - {"text1", "text2"})
            if meta_prefixes:
                print(
                    "[debug collator][WARNING] Кроме text1/text2 токенизируются дополнительные колонки. "
                    f"В loss попадут: {meta_prefixes}. Это подтверждает причину падения качества."
                )
            else:
                print("[debug collator] Только text1/text2 токенизируются (как в baseline).")

            self._debug_collator_printed = True

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


class MetaIgnoringSentenceTransformerDataCollator:
    """
    Важная фиксация: SentenceTransformerDataCollator токенизирует ВСЕ string-колонки датасета.
    При иерархических строках в датасете есть meta-колонки (doc_id/leaf/parent/...), и они превращаются
    в дополнительные sentence_features для loss (что ухудшает качество).

    Этот wrapper оставляет для loss только `text1`/`text2` (и опциональные label/dataset_name).
    """

    def __init__(
        self,
        inner: SentenceTransformerDataCollator,
        *,
        allowed_text_columns: frozenset[str] = frozenset({"text1", "text2"}),
    ) -> None:
        self.inner = inner
        self.allowed_text_columns = allowed_text_columns
        # Transformers/SentenceTransformers ожидают, что у collator есть valid_label_columns.
        # Иначе падаем в trainer.get_batch_sampler(...).
        self.valid_label_columns = list(getattr(inner, "valid_label_columns", []) or [])
        self.label_columns = set(getattr(inner, "valid_label_columns", None) or [])
        self._dataset_name_key = "dataset_name"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        filtered: list[dict[str, Any]] = []
        for f in features:
            nf: dict[str, Any] = {}
            for k, v in f.items():
                if k in self.allowed_text_columns:
                    nf[k] = v
                elif k == self._dataset_name_key:
                    nf[k] = v
                elif k in self.label_columns:
                    nf[k] = v
            filtered.append(nf)
        return self.inner(filtered)


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
    """R@20/P@20 (leaf) и R@20 (parent) на тесте через full pipeline."""
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
    from lib.eval_metrics import precision_at_k, recall_at_k
    from lib.gold_io import read_gold_csv
    from lib.grnti_ontology import load_ontology_code_map
    from tqdm import tqdm

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
        return {"R@20": 0.0, "P@20": 0.0, "R@20_parent": 0.0}

    k = EVAL_K
    recalls_leaf: List[float] = []
    precisions_leaf: List[float] = []
    recalls_parent: List[float] = []

    for it in tqdm(items, desc="Test (pipeline)", unit="doc"):
        text = it.text or ""
        if not text:
            continue
        gold_leaf = sorted(set(it.gold_codes))
        if not gold_leaf:
            continue
        gold_parent = sorted({f"{c.split('.')[0]}.{c.split('.')[1]}" for c in gold_leaf if len(c.split(".")) >= 2})
        anns = annotator.annotate(
            text=text,
            threshold=DEFAULT_THRESHOLD,
            top_k=DEFAULT_TOP_K,
            max_segment_length_for_context=DEFAULT_MAX_SEGMENT_LENGTH_FOR_CONTEXT,
            rerank_top_k=DEFAULT_RERANK_TOP_K,
            confidence_aggregation=DEFAULT_CONFIDENCE_AGGREGATION,
            filter_segments=DEFAULT_FILTER_SEGMENTS,
            use_cross_encoder_doc_score=False,
        )
        pred_leaf: List[str] = []
        for ann in anns:
            cid = ann.get("competency_id")
            if not isinstance(cid, str):
                continue
            code = competency_id_to_code.get(cid)
            if not code:
                continue
            parts = code.split(".")
            if len(parts) != 3:
                continue
            if code not in pred_leaf:
                pred_leaf.append(code)
            if len(pred_leaf) >= k:
                break
        pred_parent = []
        seen_parent = set()
        for c in pred_leaf:
            parts = c.split(".")
            if len(parts) >= 2:
                p = f"{parts[0]}.{parts[1]}"
                if p not in seen_parent:
                    seen_parent.add(p)
                    pred_parent.append(p)
        recalls_leaf.append(recall_at_k(pred_leaf, gold_leaf, k))
        precisions_leaf.append(precision_at_k(pred_leaf, gold_leaf, k))
        recalls_parent.append(recall_at_k(pred_parent, gold_parent, k))

    n = len(recalls_leaf)
    if n == 0:
        return {"R@20": 0.0, "P@20": 0.0, "R@20_parent": 0.0}
    return {
        "R@20": sum(recalls_leaf) / n,
        "P@20": sum(precisions_leaf) / n,
        "R@20_parent": sum(recalls_parent) / n,
    }


class SaveBestToDirCallback(TrainerCallback):
    """При новом лучшем eval_loss сохраняет модель в output_dir/best. В корне остаётся последняя."""

    def __init__(self, best_dir: Path, metric: str = "eval_loss", greater_is_better: bool = False):
        self.best_dir = Path(best_dir)
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.best_value = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric not in metrics:
            return control
        value = metrics[self.metric]
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return control
        is_better = (
            self.best_value is None
            or (self.greater_is_better and value > self.best_value)
            or (not self.greater_is_better and value < self.best_value)
        )
        if is_better:
            self.best_value = value
            trainer = getattr(self, "trainer", None)
            if trainer is not None and state.is_world_process_zero:
                self.best_dir.mkdir(parents=True, exist_ok=True)
                trainer.save_model(str(self.best_dir))
                print(f"  [best] {self.metric}={value:.4f} -> {self.best_dir}")
        return control


def _parent_grand_from_leaf(leaf: str) -> tuple[str, str]:
    parts = leaf.split(".")
    if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
        return f"{parts[0]}.{parts[1]}", parts[0]
    if len(parts) >= 1:
        return leaf, parts[0]
    return leaf, leaf


def build_hierarchical_rows_from_segments(
    segments_csv: Path,
    code_to_text: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Пары (сегмент, код) с метаданными для HierarchicalGrntiBatchSampler.
    doc_gold_leaves — объединение всех leaf-кодов документа по всем сегментам (multi-label safety).
    """
    import csv

    if not segments_csv.exists():
        raise FileNotFoundError(f"segments CSV not found: {segments_csv}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    raw_rows: List[Dict[str, str]] = []
    with segments_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(dict(row))

    doc_to_leaves: Dict[str, set[str]] = defaultdict(set)
    for row in raw_rows:
        doc_id = (row.get("doc_id") or "").strip()
        if not doc_id:
            continue
        codes_raw = (row.get("grnti_codes") or "").strip()
        if not codes_raw:
            continue
        codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
        codes = [c for c in codes if _is_leaf_grnti_code(c)]
        for c in codes:
            doc_to_leaves[doc_id].add(c)

    out: List[Dict[str, Any]] = []
    for row in raw_rows:
        doc_id = (row.get("doc_id") or "").strip()
        segment_text = (row.get("segment_text") or "").strip()
        if not doc_id or not segment_text:
            continue
        codes_raw = (row.get("grnti_codes") or "").strip()
        if not codes_raw:
            continue
        codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
        codes = [c for c in codes if _is_leaf_grnti_code(c)]
        if not codes:
            continue
        gold_leaves = doc_to_leaves.get(doc_id, set())
        gold_str = ";".join(sorted(gold_leaves))
        for code in codes:
            comp_text = code_to_text.get(code)
            if not comp_text:
                continue
            parent, grand = _parent_grand_from_leaf(code)
            out.append(
                {
                    "text1": segment_text,
                    "text2": comp_text,
                    "doc_id": doc_id,
                    "leaf": code,
                    "parent": parent,
                    "grand": grand,
                    "doc_gold_leaves": gold_str,
                }
            )
    return out


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
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча; с CachedMultipleNegativesRankingLoss можно ставить 64–128 и выше")
    parser.add_argument("--mini-batch-size", type=int, default=32, help="Мини-батч для cached loss (память под один forward)")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Макс. сэмплов за один запуск; с --resume следующий чанк (режим по чанкам). Без него — обучение на всех данных с чекпоинтами по шагам.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Сохранять чекпоинт каждые N шагов (при обучении на всех данных). По умолчанию 500.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-hierarchical-sampler",
        action="store_true",
        help="Иерархический sampler (far/mid/hard, doc/leaf уникальность, multi-label, safe-hard по guide)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=("gist", "cached_mnr"),
        default="cached_mnr",
        help="gist: CachedGISTEmbedLoss + замороженный guide; cached_mnr: CachedMNR (с --use-hierarchical-sampler — safe-hard остаётся в sampler)",
    )
    parser.add_argument(
        "--guide-model",
        type=str,
        default=BASE_MODEL,
        help="Замороженная guide-модель (deepvk/USER-bge-m3): для GIST и для safe-hard в sampler",
    )
    parser.add_argument(
        "--disable-guide-safe-hard",
        action="store_true",
        help="Для иерархического sampler отключить guide/safe-hard (остаются curriculum + баланс + doc/leaf + multi-label).",
    )
    parser.add_argument(
        "--gist-relative-margin",
        type=float,
        default=0.05,
        help="Relative margin для CachedGISTEmbedLoss и для safe-hard в sampler; типичная сетка: 0.03 / 0.05 / 0.08 / 0.10",
    )
    parser.add_argument(
        "--curriculum-epoch1",
        type=str,
        default="0.8,0.2,0",
        help="Доли far,mid,hard для эпохи 1 (стабилизация), через запятую",
    )
    parser.add_argument(
        "--curriculum-epoch2",
        type=str,
        default="0.6,0.3,0.1",
        help="Доли far,mid,hard для эпохи 2",
    )
    parser.add_argument(
        "--curriculum-epoch3plus",
        type=str,
        default="0.45,0.35,0.2",
        help="Доли far,mid,hard для эпохи 3 и далее",
    )
    parser.add_argument(
        "--leaf-balance-power",
        type=float,
        default=0.5,
        help="Степень оверсэмплинга редких leaf (чем больше, тем сильнее хвост)",
    )
    parser.add_argument(
        "--grand-balance-weight",
        type=float,
        default=1.0,
        help="Вес баланса по разделу grand (grand с низкой частотой — выше вес)",
    )
    parser.add_argument(
        "--max-scored-candidates",
        type=int,
        default=256,
        help="Сколько кандидатов оценивать scoring-ом curriculum на шаг добавления в батч",
    )
    parser.add_argument(
        "--no-sampler-fallback-relaxed",
        action="store_true",
        help="Не ослаблять выбор до «первого basic+multilabel+safe» при плохом score",
    )
    parser.add_argument(
        "--no-sampler-diagnostics",
        action="store_true",
        help="Не печатать сводку иерархического sampler в конце эпохи",
    )
    parser.add_argument(
        "--precomputed-batches",
        type=str,
        default="",
        help="Путь к .pt от generate_hierarchical_batches.py: те же батчи по эпохам без онлайн-сборки",
    )
    parser.add_argument(
        "--dataloader-drop-last",
        action="store_true",
        help="Должен совпадать с --drop-last при генерации предбатчей (иначе factory выдаст ошибку)",
    )
    parser.add_argument(
        "--skip-baseline-test",
        action="store_true",
        help="Не гонять тест на base_model (только finetuned → result.json). Удобно для Optuna / экономии времени",
    )
    parser.add_argument(
        "--debug-collator-meta-tokenization",
        action="store_true",
        help="Один раз печатает какие колонки кроме text1/text2 токенизируются collator'ом и уходят в loss.",
    )
    args = parser.parse_args()
    run_training(args)

def run_training(args) -> Dict[str, Any]:
    skip_baseline_test = bool(getattr(args, "skip_baseline_test", False))
    debug_collator_meta_tokenization = bool(getattr(args, "debug_collator_meta_tokenization", False))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    base_output_dir = Path(args.output_dir)
    resume_path = Path(args.resume.strip()) if args.resume.strip() else None
    checkpoint_dirs = []
    if resume_path and resume_path.is_dir():
        for p in resume_path.iterdir():
            if p.is_dir() and p.name.startswith("checkpoint-"):
                try:
                    step = int(p.name.split("-")[1])
                    checkpoint_dirs.append((step, p))
                except (IndexError, ValueError):
                    pass
        checkpoint_dirs.sort(key=lambda x: x[0])
    resume_from_checkpoint = None
    if checkpoint_dirs:
        resume_from_checkpoint = str(checkpoint_dirs[-1][1])
        print(f"Найден чекпоинт: {resume_from_checkpoint}, продолжение с этого шага")

    samples_done = 0
    use_chunked = args.max_train_samples is not None and args.max_train_samples > 0

    if resume_from_checkpoint:
        model = SentenceTransformer(resume_from_checkpoint, device=device)
        print("Модель загружена из чекпоинта, обучение продолжится с этого шага")
    elif args.resume.strip() and use_chunked:
        model = SentenceTransformer(str(resume_path), device=device)
        print(f"Resumed from: {args.resume}")
        state_path = resume_path / "training_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            samples_done = int(state.get("samples_done", 0))
            print(f"Resuming from samples_done={samples_done} (следующий чанк)")
    elif args.resume.strip():
        model = SentenceTransformer(str(resume_path), device=device)
        print(f"Resumed from: {args.resume}")
        print("В каталоге нет чекпоинтов (checkpoint-*). Обучение начнётся с шага 0 (полные эпохи).")
    else:
        model = SentenceTransformer(args.base_model, device=device)

    code_to_text = load_ontology_texts(ONTOLOGY_PATH)
    use_precomputed_batches = bool(args.precomputed_batches.strip())
    if use_precomputed_batches and args.use_hierarchical_sampler:
        raise ValueError("Нельзя одновременно --precomputed-batches и --use-hierarchical-sampler")

    if args.use_hierarchical_sampler or use_precomputed_batches:
        train_rows = build_hierarchical_rows_from_segments(TRAIN_SEGMENTS_CSV, code_to_text)
        if not train_rows:
            raise RuntimeError("no hierarchical training rows built")
        mode = "предбатчи" if use_precomputed_batches else "иерархический sampler"
        print(f"Train rows ({mode}): {len(train_rows)}")
        full_train_dataset = Dataset.from_list(train_rows).shuffle(seed=int(args.seed))
    else:
        train_examples = build_pairs_from_segments(TRAIN_SEGMENTS_CSV, code_to_text)
        if not train_examples:
            raise RuntimeError("no training pairs built")
        print(f"Train pairs (из сегментов): {len(train_examples)}")
        full_train_dataset = Dataset.from_list(
            [{"text1": ex.texts[0], "text2": ex.texts[1]} for ex in train_examples]
        ).shuffle(seed=int(args.seed))

    valid_examples = build_pairs_from_segments(VALID_SEGMENTS_CSV, code_to_text)
    if not valid_examples:
        raise RuntimeError("no validation pairs built")
    print(f"Valid pairs (из сегментов): {len(valid_examples)}")
    valid_dataset = Dataset.from_list([{"text1": ex.texts[0], "text2": ex.texts[1]} for ex in valid_examples])

    n = len(full_train_dataset)
    max_n = int(args.max_train_samples) if args.max_train_samples is not None else None
    if max_n is not None and max_n <= 0:
        raise ValueError("--max-train-samples must be positive")

    if resume_from_checkpoint:
        train_dataset = full_train_dataset
        print(f"Train dataset: {len(train_dataset)} сэмплов (продолжение с чекпоинта)")
    elif args.resume.strip() and use_chunked:
        start = samples_done
        end = min(samples_done + max_n, n)
        if start >= n or start >= end:
            print("Нет новых сэмплов для этого чанка. Выход.")
            return {"status": "no_new_samples"}
        train_dataset = full_train_dataset.select(range(start, end))
        print(f"Чанк для этого запуска: сэмплы {start}–{end} (всего {len(train_dataset)})")
    elif max_n is not None:
        train_dataset = full_train_dataset.select(range(0, min(max_n, n)))
        print(f"Train dataset (первый чанк): {len(train_dataset)} сэмплов")
    else:
        train_dataset = full_train_dataset
        print(f"Train dataset: {len(train_dataset)} сэмплов (чекпоинты каждые {args.save_steps} шагов)")

    # Для precomputed-batches meta-колонки не нужны на этапе обучения:
    # precomputed sampler использует только индексы, а loss обучается на text1/text2.
    # Это убирает риск "лишнего текста" и позволяет использовать обычный collator.
    if use_precomputed_batches:
        to_remove = sorted(_HIERARCHICAL_SAMPLER_META_COLS & set(train_dataset.column_names))
        if to_remove:
            train_dataset = train_dataset.remove_columns(to_remove)
            print(f"Precomputed mode: удалены meta-колонки из train_dataset: {to_remove}")

    if args.use_hierarchical_sampler:
        _hier_cols = {"text1", "text2", "doc_id", "leaf", "parent", "grand", "doc_gold_leaves"}
        _missing = _hier_cols - set(train_dataset.column_names)
        if _missing:
            raise RuntimeError(
                f"Датасет не содержит колонок для иерархического sampler: {sorted(_missing)}. "
                f"Текущие колонки: {list(train_dataset.column_names)}. "
                "Нужны строки из build_hierarchical_rows_from_segments (сегменты с doc_id, grnti_codes). "
                "Проверьте: при запуске указан --use-hierarchical-sampler; TRAIN_SEGMENTS_CSV не пустой и "
                "совпадает с ожидаемым путём; не подменяйте train_dataset на пары только с text1/text2."
            )

    if args.loss == "gist" and args.disable_guide_safe_hard:
        raise ValueError("--disable-guide-safe-hard несовместим с --loss gist (GIST требует guide).")
    use_guide_for_sampler = bool(args.use_hierarchical_sampler and not args.disable_guide_safe_hard)
    guide_st: SentenceTransformer | None = None
    if args.loss == "gist" or use_guide_for_sampler:
        guide_st = SentenceTransformer(args.guide_model, device=device)
        guide_st.eval()
        for p in guide_st.parameters():
            p.requires_grad = False
        print(f"Guide model (frozen): {args.guide_model}")

    if args.loss == "gist":
        assert guide_st is not None
        train_loss = losses.CachedGISTEmbedLoss(
            model,
            guide_st,
            mini_batch_size=int(args.mini_batch_size),
            show_progress_bar=False,
            margin_strategy="relative",
            margin=float(args.gist_relative_margin),
        )
        print(f"Loss: CachedGISTEmbedLoss (relative margin={args.gist_relative_margin})")
    else:
        train_loss = losses.CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=int(args.mini_batch_size), show_progress_bar=False
        )
        print("Loss: CachedMultipleNegativesRankingLoss")

    output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    use_save_steps = not use_chunked
    save_steps_val = int(args.save_steps) if use_save_steps else 500

    if use_precomputed_batches:
        from lib.precomputed_epoch_batch_sampler import create_precomputed_batch_sampler_factory

        batch_sampler_arg = create_precomputed_batch_sampler_factory(Path(args.precomputed_batches.strip()))
        print(f"Batch sampler: precomputed ({args.precomputed_batches.strip()})")
    elif args.use_hierarchical_sampler:
        from lib.hierarchical_grnti_batch_sampler import create_hierarchical_batch_sampler_factory

        batch_sampler_arg = create_hierarchical_batch_sampler_factory(
            guide_model=guide_st if use_guide_for_sampler else None,
            curriculum_epoch1=args.curriculum_epoch1,
            curriculum_epoch2=args.curriculum_epoch2,
            curriculum_epoch3plus=args.curriculum_epoch3plus,
            relative_margin=float(args.gist_relative_margin),
            leaf_balance_power=float(args.leaf_balance_power),
            grand_balance_weight=float(args.grand_balance_weight),
            max_scored_candidates=int(args.max_scored_candidates),
            enable_diagnostics=not args.no_sampler_diagnostics,
            fallback_relaxed=not args.no_sampler_fallback_relaxed,
        )
    else:
        batch_sampler_arg = BatchSamplers.NO_DUPLICATES

    # Кастомный batch_sampler + num_workers>0 на Windows часто даёт pickle ошибки
    # («Can't pickle local object ...» / несериализуемый guide). Для callable — только главный процесс.
    use_custom_batch_sampler = bool(use_precomputed_batches or args.use_hierarchical_sampler)
    dataloader_workers = 0 if use_custom_batch_sampler else 2

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(args.epochs),
        per_device_train_batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        batch_sampler=batch_sampler_arg,
        dataloader_drop_last=bool(args.dataloader_drop_last),
        warmup_ratio=float(args.warmup_ratio),
        seed=int(args.seed),
        logging_steps=50,
        save_strategy="steps" if use_save_steps else "epoch",
        save_steps=save_steps_val,
        save_total_limit=2,
        eval_strategy="steps" if use_save_steps else "epoch",
        eval_steps=save_steps_val if use_save_steps else None,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=dataloader_workers,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    if use_custom_batch_sampler:
        print(f"DataLoader: dataloader_num_workers={dataloader_workers} (кастомный batch_sampler)")

    # Важно: loss должен учиться только на text1->text2.
    # Без этого SentenceTransformerDataCollator токенизирует meta-колонки
    # (doc_id/leaf/parent/...) и они попадают в sentence_features для loss.
    tokenizer = getattr(model, "tokenizer", None)
    all_special_ids = (
        set(tokenizer.all_special_ids)
        if tokenizer is not None and hasattr(tokenizer, "all_special_ids")
        else set()
    )
    base_collator = SentenceTransformerDataCollator(
        tokenize_fn=model.tokenize,
        router_mapping=getattr(training_args, "router_mapping", None) or {},
        prompts=getattr(training_args, "prompts", None) or None,
        all_special_ids=all_special_ids,
    )
    collator = base_collator if use_precomputed_batches else MetaIgnoringSentenceTransformerDataCollator(base_collator)

    trainer = FinetuneBiEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        loss=train_loss,
        callbacks=[SaveBestToDirCallback(output_dir / "best", metric="eval_loss", greater_is_better=False)],
        data_collator=collator,
        debug_collator_meta_tokenization=debug_collator_meta_tokenization,
    )
    payload: Dict[str, Any] = {}
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(str(output_dir))

        final_samples_done = samples_done + len(train_dataset) if use_chunked else None
        if use_chunked:
            (output_dir / "training_state.json").write_text(
                json.dumps({"samples_done": final_samples_done}, indent=2), encoding="utf-8"
            )

        if skip_baseline_test:
            print("Test (pipeline): пропуск baseline (--skip-baseline-test), только finetuned")
            original_metrics: Dict[str, float] = {}
        else:
            original_metrics = evaluate_on_test(args.base_model)
        finetuned_metrics = evaluate_on_test(str(output_dir))
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "result.json"
        payload = {
            "original_metrics": original_metrics,
            "finetuned_metrics": finetuned_metrics,
            "skip_baseline_test": skip_baseline_test,
            "result_path": str(result_path),
            "params": {
            "base_model": args.base_model,
            "output_dir": str(output_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "save_steps": args.save_steps,
            "max_train_samples": args.max_train_samples,
            "samples_done": final_samples_done,
            "seed": args.seed,
            "use_hierarchical_sampler": args.use_hierarchical_sampler,
            "loss": args.loss,
            "guide_model": args.guide_model,
            "disable_guide_safe_hard": bool(args.disable_guide_safe_hard),
            "gist_relative_margin": args.gist_relative_margin,
            "curriculum_epoch1": args.curriculum_epoch1,
            "curriculum_epoch2": args.curriculum_epoch2,
            "curriculum_epoch3plus": args.curriculum_epoch3plus,
            "leaf_balance_power": args.leaf_balance_power,
            "grand_balance_weight": args.grand_balance_weight,
            "max_scored_candidates": args.max_scored_candidates,
            "precomputed_batches": args.precomputed_batches.strip() or None,
            "dataloader_drop_last": bool(args.dataloader_drop_last),
            "skip_baseline_test": skip_baseline_test,
            },
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved run result to: {result_path}")
    finally:
        # Для Optuna/in-process запусков: явная очистка CUDA-памяти между trial'ами.
        del trainer
        del train_loss
        if guide_st is not None:
            del guide_st
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    return payload


if __name__ == "__main__":
    main()

