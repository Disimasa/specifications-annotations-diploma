from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

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

from lib.hierarchical_grnti_batch_sampler import create_hierarchical_batch_sampler_factory, relation_type
from scripts.train.finetune_bi_encoder import (
    BASE_MODEL,
    ONTOLOGY_PATH,
    TRAIN_SEGMENTS_CSV,
    build_hierarchical_rows_from_segments,
    load_ontology_texts,
)


@dataclass
class RunConfig:
    max_scored_candidates: int
    leaf_balance_power: float
    grand_balance_weight: float
    relative_margin: float
    fallback_relaxed: bool


def _parse_int_grid(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_grid(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _iter_configs(args) -> Iterable[RunConfig]:
    for msc, lbp, gbw, rm in itertools.product(
        _parse_int_grid(args.max_scored_candidates_grid),
        _parse_float_grid(args.leaf_balance_power_grid),
        _parse_float_grid(args.grand_balance_weight_grid),
        _parse_float_grid(args.relative_margin_grid),
    ):
        yield RunConfig(
            max_scored_candidates=msc,
            leaf_balance_power=lbp,
            grand_balance_weight=gbw,
            relative_margin=rm,
            fallback_relaxed=not args.no_sampler_fallback_relaxed,
        )


def _analyze_batches(dataset: Dataset, batches: List[List[int]]) -> Dict[str, float]:
    e_far = e_mid = e_hard = 0
    dup_doc_viol = dup_leaf_viol = 0
    for batch in batches:
        docs = set()
        leaves = set()
        for i in batch:
            row = dataset[i]
            doc = str(row["doc_id"])
            leaf = str(row["leaf"])
            if doc in docs:
                dup_doc_viol += 1
            docs.add(doc)
            if leaf in leaves:
                dup_leaf_viol += 1
            leaves.add(leaf)

        for ii in range(len(batch)):
            for jj in range(len(batch)):
                if ii == jj:
                    continue
                a = dataset[batch[ii]]
                b = dataset[batch[jj]]
                t = relation_type(
                    str(a["leaf"]),
                    str(a["grand"]),
                    str(a["parent"]),
                    str(b["leaf"]),
                    str(b["grand"]),
                    str(b["parent"]),
                )
                if t == "far":
                    e_far += 1
                elif t == "mid":
                    e_mid += 1
                elif t == "hard":
                    e_hard += 1

    tot_edges = e_far + e_mid + e_hard
    return {
        "batches": float(len(batches)),
        "avg_batch_size": float(sum(len(b) for b in batches) / max(1, len(batches))),
        "edge_far_frac": float(e_far / tot_edges) if tot_edges else 0.0,
        "edge_mid_frac": float(e_mid / tot_edges) if tot_edges else 0.0,
        "edge_hard_frac": float(e_hard / tot_edges) if tot_edges else 0.0,
        "dup_doc_violations": float(dup_doc_viol),
        "dup_leaf_violations": float(dup_leaf_viol),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Быстрый playground для иерархического батчера на маленькой выборке.")
    parser.add_argument("--segments-csv", type=str, default=str(TRAIN_SEGMENTS_CSV))
    parser.add_argument("--ontology-path", type=str, default=str(ONTOLOGY_PATH))
    parser.add_argument("--guide-model", type=str, default=BASE_MODEL)
    parser.add_argument("--disable-guide-safe-hard", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=4096, help="Сколько строк взять в playground после shuffle")
    parser.add_argument("--max-batches", type=int, default=120, help="Сколько батчей максимум анализировать на эпоху")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--curriculum-epoch1", type=str, default="0.8,0.2,0")
    parser.add_argument("--curriculum-epoch2", type=str, default="0.6,0.3,0.1")
    parser.add_argument("--curriculum-epoch3plus", type=str, default="0.45,0.35,0.2")
    parser.add_argument("--max-scored-candidates-grid", type=str, default="64,128,256")
    parser.add_argument("--leaf-balance-power-grid", type=str, default="0.5")
    parser.add_argument("--grand-balance-weight-grid", type=str, default="1.0")
    parser.add_argument("--relative-margin-grid", type=str, default="0.05")
    parser.add_argument("--no-sampler-fallback-relaxed", action="store_true")
    parser.add_argument("--save-report", type=str, default="", help="Путь для JSON-отчета")
    args = parser.parse_args()

    code_to_text = load_ontology_texts(Path(args.ontology_path))
    rows = build_hierarchical_rows_from_segments(Path(args.segments_csv), code_to_text)
    if not rows:
        raise RuntimeError("Не удалось собрать строки для иерархического датасета.")

    dataset = Dataset.from_list(rows).shuffle(seed=int(args.seed))
    if args.sample_size > 0:
        dataset = dataset.select(range(0, min(int(args.sample_size), len(dataset))))
    print(f"Playground dataset size: {len(dataset)}")

    guide_st = None
    if not args.disable_guide_safe_hard:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        guide_st = SentenceTransformer(args.guide_model, device=device)
        guide_st.eval()
        for p in guide_st.parameters():
            p.requires_grad = False
        print(f"Guide enabled: {args.guide_model}")
    else:
        print("Guide disabled (safe-hard off)")

    configs = list(_iter_configs(args))
    if not configs:
        raise ValueError("Пустая сетка конфигов")
    print(f"Configs to run: {len(configs)}")

    all_results: List[Dict[str, float | int | str]] = []
    for ci, cfg in enumerate(tqdm(configs, desc="Configs", unit="cfg"), start=1):
        sampler_factory = create_hierarchical_batch_sampler_factory(
            guide_model=guide_st,
            curriculum_epoch1=args.curriculum_epoch1,
            curriculum_epoch2=args.curriculum_epoch2,
            curriculum_epoch3plus=args.curriculum_epoch3plus,
            relative_margin=cfg.relative_margin,
            leaf_balance_power=cfg.leaf_balance_power,
            grand_balance_weight=cfg.grand_balance_weight,
            max_scored_candidates=cfg.max_scored_candidates,
            enable_diagnostics=False,
            fallback_relaxed=cfg.fallback_relaxed,
        )
        sampler = sampler_factory(
            dataset=dataset,
            batch_size=int(args.batch_size),
            drop_last=bool(args.drop_last),
            valid_label_columns=None,
            generator=torch.Generator(),
            seed=int(args.seed),
        )

        for ep0 in range(int(args.epochs)):
            sampler.set_epoch(ep0)
            t0 = time.perf_counter()
            t_first = None
            batches: List[List[int]] = []
            for b in sampler:
                if t_first is None:
                    t_first = time.perf_counter()
                batches.append(list(b))
                if args.max_batches > 0 and len(batches) >= int(args.max_batches):
                    break
            t1 = time.perf_counter()
            stat = _analyze_batches(dataset, batches)
            row: Dict[str, float | int | str] = {
                "config_id": ci,
                "epoch": ep0 + 1,
                "max_scored_candidates": cfg.max_scored_candidates,
                "leaf_balance_power": cfg.leaf_balance_power,
                "grand_balance_weight": cfg.grand_balance_weight,
                "relative_margin": cfg.relative_margin,
                "fallback_relaxed": int(cfg.fallback_relaxed),
                "batches_collected": len(batches),
                "time_total_sec": float(t1 - t0),
                "time_to_first_batch_sec": float((t_first - t0) if t_first is not None else -1.0),
                **stat,
            }
            all_results.append(row)
            print(
                f"[cfg {ci}/{len(configs)} ep {ep0+1}] "
                f"t_first={row['time_to_first_batch_sec']:.2f}s total={row['time_total_sec']:.2f}s "
                f"batches={row['batches_collected']} "
                f"far/mid/hard={row['edge_far_frac']:.2f}/{row['edge_mid_frac']:.2f}/{row['edge_hard_frac']:.2f}"
            )

    all_results_sorted = sorted(
        all_results,
        key=lambda x: (float(x["time_total_sec"]), -float(x["edge_hard_frac"])),
    )
    print("\nTop-5 fastest configs:")
    for r in all_results_sorted[:5]:
        print(json.dumps(r, ensure_ascii=False))

    if args.save_report.strip():
        out = Path(args.save_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results_sorted, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report: {out}")


if __name__ == "__main__":
    main()

