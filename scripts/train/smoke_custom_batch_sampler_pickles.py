"""
Быстрая проверка без Optuna: picklable фабрики batch_sampler + создание HierarchicalGrntiBatchSampler.

Запуск из корня репозитория:
  python scripts/train/smoke_custom_batch_sampler_pickles.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
for p in (str(PROJECT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets import Dataset

from lib.hierarchical_grnti_batch_sampler import HierarchicalGrntiBatchSamplerFactory


def main() -> None:
    rows = [
        {
            "text1": "сегмент один",
            "text2": "компетенция a",
            "doc_id": "d1",
            "leaf": "1.2.3",
            "parent": "1.2",
            "grand": "1",
            "doc_gold_leaves": "1.2.3",
        },
        {
            "text1": "сегмент два",
            "text2": "компетенция b",
            "doc_id": "d2",
            "leaf": "2.3.4",
            "parent": "2.3",
            "grand": "2",
            "doc_gold_leaves": "2.3.4",
        },
    ]
    ds = Dataset.from_list(rows)

    factory = HierarchicalGrntiBatchSamplerFactory(
        guide_model=None,
        curriculum_epoch1="random",
        curriculum_epoch2="0.6,0.3,0.1",
        curriculum_epoch3plus="0.45,0.35,0.2",
        relative_margin=0.05,
        leaf_balance_power=0.5,
        grand_balance_weight=1.0,
        max_scored_candidates=64,
        enable_diagnostics=False,
        fallback_relaxed=True,
    )

    blob = pickle.dumps(factory)
    factory2 = pickle.loads(blob)
    assert isinstance(factory2, HierarchicalGrntiBatchSamplerFactory)

    sampler = factory2(ds, batch_size=2, drop_last=False, valid_label_columns=["label"], seed=0)
    _ = len(sampler)
    n = sum(1 for _ in sampler)
    print(f"pickle hierarchical factory: OK; sampler batches (epoch 0): {n}")


if __name__ == "__main__":
    main()
