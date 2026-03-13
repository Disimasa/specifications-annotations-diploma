from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[2]

TRAIN_IN = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_train_augmented.csv"
TEST_IN = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test.csv"

TRAIN_OUT = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_train_augmented_clean.csv"
VALID_OUT = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_valid.csv"


def _set_field_size_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)


def _key(row: Dict[str, str]) -> Tuple[str, str]:
    title = (row.get("title") or "").strip().lower()
    abstract = (row.get("abstract") or "").strip().lower()
    return title, abstract


def _load_keys(path: Path) -> set[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    keys: set[Tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = _key(row)
            if not k[0] and not k[1]:
                continue
            keys.add(k)
    return keys


def _read_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(row)
    if not fieldnames:
        raise RuntimeError(f"No headers found in CSV: {path}")
    return rows, fieldnames


def _write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(encoding="utf-8", newline="", mode="w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    _set_field_size_limit()

    test_keys = _load_keys(TEST_IN)

    train_rows, fieldnames = _read_rows(TRAIN_IN)
    before = len(train_rows)

    cleaned: List[Dict[str, str]] = []
    removed = 0
    for row in train_rows:
        k = _key(row)
        if (k[0] or k[1]) and k in test_keys:
            removed += 1
            continue
        cleaned.append(row)

    if not cleaned:
        raise RuntimeError("После удаления пересечений train стал пустым.")

    rnd = random.Random(42)
    rnd.shuffle(cleaned)

    valid_n = max(1, int(round(0.10 * len(cleaned))))
    valid_rows = cleaned[:valid_n]
    train_clean_rows = cleaned[valid_n:]

    _write_rows(TRAIN_OUT, train_clean_rows, fieldnames)
    _write_rows(VALID_OUT, valid_rows, fieldnames)

    print(f"Train before: {before}")
    print(f"Removed overlaps with test: {removed}")
    print(f"Train clean: {len(train_clean_rows)} -> {TRAIN_OUT}")
    print(f"Valid (10%): {len(valid_rows)} -> {VALID_OUT}")


if __name__ == "__main__":
    main()

