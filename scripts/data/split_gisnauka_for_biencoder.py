from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_DIR / "data" / "gold" / "gisnauka_samples.csv"
TEST_PATH = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test.csv"
TRAIN_PATH = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_train.csv"


def _is_leaf_grnti_code(code: str) -> bool:
    parts = code.split(".")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


def split() -> Tuple[int, int]:
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Источник не найден: {SRC_PATH}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    with SRC_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = list(reader)
        fieldnames = reader.fieldnames or []

    eligible_indices: List[int] = []
    effective_top_codes: List[str] = []

    for i, row in enumerate(rows):
        title = (row.get("title") or "").strip()
        abstract = (row.get("abstract") or "").strip()
        codes_raw = (row.get("grnti_codes") or "").strip()
        if not codes_raw:
            effective_top_codes.append("")
            continue
        codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
        codes = [c for c in codes if _is_leaf_grnti_code(c)]
        if not codes:
            effective_top_codes.append("")
            continue
        text = f"{title}\n\n{abstract}".strip()
        if not text:
            effective_top_codes.append("")
            continue
        top_code = (row.get("top_code") or "").strip() or codes[0].split(".")[0]
        if not top_code:
            effective_top_codes.append("")
            continue
        eligible_indices.append(i)
        effective_top_codes.append(top_code)

    by_top: Dict[str, List[int]] = {}
    for idx in eligible_indices:
        top = effective_top_codes[idx]
        if not top:
            continue
        by_top.setdefault(top, []).append(idx)

    test_indices: List[int] = []
    for top in sorted(by_top.keys()):
        group = by_top[top]
        if not group:
            continue
        test_indices.append(group[0])

    test_set = set(test_indices)

    test_rows: List[Dict[str, str]] = [rows[i] for i in sorted(test_indices)]
    train_rows: List[Dict[str, str]] = [
        rows[i] for i in range(len(rows)) if i in eligible_indices and i not in test_set
    ]

    if not fieldnames:
        fieldnames = list(rows[0].keys()) if rows else []

    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TEST_PATH.open("w", encoding="utf-8", newline="") as f_test:
        writer = csv.DictWriter(f_test, fieldnames=fieldnames)
        writer.writeheader()
        for r in test_rows:
            writer.writerow(r)

    with TRAIN_PATH.open("w", encoding="utf-8", newline="") as f_train:
        writer = csv.DictWriter(f_train, fieldnames=fieldnames)
        writer.writeheader()
        for r in train_rows:
            writer.writerow(r)

    return len(test_rows), len(train_rows)


def main() -> None:
    n_test, n_train = split()
    print(f"test rows: {n_test}")
    print(f"train rows: {n_train}")
    print(f"test csv:  {TEST_PATH}")
    print(f"train csv: {TRAIN_PATH}")


if __name__ == "__main__":
    main()

