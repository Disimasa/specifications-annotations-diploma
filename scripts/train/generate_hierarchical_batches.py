from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lib.hierarchical_grnti_batch_sampler import create_hierarchical_batch_sampler_factory
from scripts.train.finetune_bi_encoder import (
    BASE_MODEL,
    ONTOLOGY_PATH,
    TRAIN_SEGMENTS_CSV,
    build_hierarchical_rows_from_segments,
    load_ontology_texts,
)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-").lower()


def _fmt_float(v: float) -> str:
    return f"{float(v):.4f}".rstrip("0").rstrip(".")


def _auto_filename(args: Any) -> str:
    guide_tag = _slug(str(args.guide_model).split("/")[-1])
    seg_tag = _slug(Path(args.segments_csv).stem)
    c1 = _slug(args.curriculum_epoch1.replace(",", "-"))
    c2 = _slug(args.curriculum_epoch2.replace(",", "-"))
    c3 = _slug(args.curriculum_epoch3plus.replace(",", "-"))
    cfg = {
        "segments_csv": str(args.segments_csv),
        "ontology_path": str(args.ontology_path),
        "guide_model": args.guide_model,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "relative_margin": float(args.relative_margin),
        "curriculum_epoch1": args.curriculum_epoch1,
        "curriculum_epoch2": args.curriculum_epoch2,
        "curriculum_epoch3plus": args.curriculum_epoch3plus,
        "leaf_balance_power": float(args.leaf_balance_power),
        "grand_balance_weight": float(args.grand_balance_weight),
        "max_scored_candidates": int(args.max_scored_candidates),
        "fallback_relaxed": bool(not args.no_sampler_fallback_relaxed),
        "disable_guide_safe_hard": bool(args.disable_guide_safe_hard),
        "max_train_samples": args.max_train_samples,
    }
    digest = hashlib.sha1(json.dumps(cfg, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:10]
    return (
        f"hb_{seg_tag}_bs{args.batch_size}_ep{args.epochs}_seed{args.seed}_"
        f"m{_fmt_float(args.relative_margin)}_guide-{guide_tag}_"
        f"c1-{c1}_c2-{c2}_c3-{c3}_{digest}.pt"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Предгенерация батчей HierarchicalGrntiBatchSampler по эпохам с сохранением в .pt"
    )
    parser.add_argument("--segments-csv", type=str, default=str(TRAIN_SEGMENTS_CSV))
    parser.add_argument("--ontology-path", type=str, default=str(ONTOLOGY_PATH))
    parser.add_argument("--guide-model", type=str, default=BASE_MODEL)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_DIR / "data" / "gold" / "precomputed_batches"))
    parser.add_argument(
        "--output-name",
        type=str,
        default="",
        help="Имя выходного файла .pt. Если не задано, будет авто-имя по аргументам.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Для скольких эпох предгенерировать батчи")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--relative-margin", type=float, default=0.05)
    parser.add_argument("--curriculum-epoch1", type=str, default="0.8,0.2,0")
    parser.add_argument("--curriculum-epoch2", type=str, default="0.6,0.3,0.1")
    parser.add_argument("--curriculum-epoch3plus", type=str, default="0.45,0.35,0.2")
    parser.add_argument("--leaf-balance-power", type=float, default=0.5)
    parser.add_argument("--grand-balance-weight", type=float, default=1.0)
    parser.add_argument("--max-scored-candidates", type=int, default=256)
    parser.add_argument("--no-sampler-fallback-relaxed", action="store_true")
    parser.add_argument(
        "--disable-guide-safe-hard",
        action="store_true",
        help="Отключить guide/safe-hard при генерации (сохраняются остальные правила иерархического sampler).",
    )
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--disable-sampler-diagnostics", action="store_true")
    args = parser.parse_args()

    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_train_samples is not None and int(args.max_train_samples) <= 0:
        raise ValueError("--max-train-samples must be positive")

    segments_csv = Path(args.segments_csv)
    ontology_path = Path(args.ontology_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ontology and building hierarchical rows...")
    code_to_text = load_ontology_texts(ontology_path)
    rows = build_hierarchical_rows_from_segments(segments_csv, code_to_text)
    if not rows:
        raise RuntimeError("No rows built for hierarchical sampler")

    dataset = Dataset.from_list(rows).shuffle(seed=int(args.seed))
    if args.max_train_samples is not None:
        dataset = dataset.select(range(0, min(int(args.max_train_samples), len(dataset))))
    print(f"Rows in dataset: {len(dataset)}")

    guide_st: SentenceTransformer | None = None
    if not args.disable_guide_safe_hard:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        guide_st = SentenceTransformer(args.guide_model, device=device)
        guide_st.eval()
        for p in guide_st.parameters():
            p.requires_grad = False
        print(f"Guide model loaded (frozen): {args.guide_model}")
    else:
        print("Guide disabled: safe-hard checks are off")

    batch_sampler_fn = create_hierarchical_batch_sampler_factory(
        guide_model=guide_st,
        curriculum_epoch1=args.curriculum_epoch1,
        curriculum_epoch2=args.curriculum_epoch2,
        curriculum_epoch3plus=args.curriculum_epoch3plus,
        relative_margin=float(args.relative_margin),
        leaf_balance_power=float(args.leaf_balance_power),
        grand_balance_weight=float(args.grand_balance_weight),
        max_scored_candidates=int(args.max_scored_candidates),
        enable_diagnostics=not args.disable_sampler_diagnostics,
        fallback_relaxed=not args.no_sampler_fallback_relaxed,
    )
    sampler = batch_sampler_fn(
        dataset=dataset,
        batch_size=int(args.batch_size),
        drop_last=bool(args.drop_last),
        valid_label_columns=None,
        generator=torch.Generator(),
        seed=int(args.seed),
    )

    batches_by_epoch: List[List[List[int]]] = []
    diag_by_epoch: List[Dict[str, float]] = []
    total_batches = 0

    for epoch0 in tqdm(range(int(args.epochs)), desc="Epochs", unit="epoch"):
        sampler.set_epoch(epoch0)
        print(f"Generating epoch {epoch0 + 1}/{args.epochs}...")
        epoch_batches = [
            list(batch)
            for batch in tqdm(
                sampler,
                total=len(sampler),
                desc=f"Epoch {epoch0 + 1} batches",
                unit="batch",
                leave=False,
            )
        ]
        batches_by_epoch.append(epoch_batches)
        diag_by_epoch.append(dict(getattr(sampler, "diagnostics_last_epoch", {})))
        total_batches += len(epoch_batches)
        print(
            f"  epoch={epoch0 + 1}: batches={len(epoch_batches)} "
            f"diag={json.dumps(diag_by_epoch[-1], ensure_ascii=False)}"
        )

    output_name = args.output_name.strip() or _auto_filename(args)
    if not output_name.endswith(".pt"):
        output_name += ".pt"
    output_path = output_dir / output_name

    payload = {
        "format_version": 1,
        "dataset_size": len(dataset),
        "batches_by_epoch": batches_by_epoch,
        "diagnostics_by_epoch": diag_by_epoch,
        "args": {
            "segments_csv": str(segments_csv),
            "ontology_path": str(ontology_path),
            "guide_model": args.guide_model,
            "disable_guide_safe_hard": bool(args.disable_guide_safe_hard),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "drop_last": bool(args.drop_last),
            "max_train_samples": args.max_train_samples,
            "relative_margin": float(args.relative_margin),
            "curriculum_epoch1": args.curriculum_epoch1,
            "curriculum_epoch2": args.curriculum_epoch2,
            "curriculum_epoch3plus": args.curriculum_epoch3plus,
            "leaf_balance_power": float(args.leaf_balance_power),
            "grand_balance_weight": float(args.grand_balance_weight),
            "max_scored_candidates": int(args.max_scored_candidates),
            "fallback_relaxed": bool(not args.no_sampler_fallback_relaxed),
        },
    }
    torch.save(payload, output_path)
    print(f"Saved precomputed batches: {output_path}")
    print(f"Total batches: {total_batches}")


if __name__ == "__main__":
    main()

