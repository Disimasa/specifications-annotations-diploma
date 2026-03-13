from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
TRAIN_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_train.csv"
TEST_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test.csv"


def load_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)
    keys: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("title") or "").strip().lower()
            abstract = (row.get("abstract") or "").strip().lower()
            if not title and not abstract:
                continue
            keys.add((title, abstract))
    return keys


def main() -> None:
    train_keys = load_keys(TRAIN_CSV)
    test_keys = load_keys(TEST_CSV)

    inter = train_keys & test_keys

    print(f"Train unique docs: {len(train_keys)}")
    print(f"Test unique docs: {len(test_keys)}")
    print(f"Overlap docs: {len(inter)}")

    if inter:
        print("Примеры пересечений (до 5):")
        for i, (title, abstract) in enumerate(sorted(inter)[:5], start=1):
            print(f"#{i}")
            print(f"TITLE: {title[:200]}")
            print(f"ABSTRACT: {abstract[:200]}")
            print("---")


if __name__ == "__main__":
    main()

