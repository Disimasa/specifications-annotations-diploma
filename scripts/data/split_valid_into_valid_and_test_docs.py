from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[2]

VALID_IN = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_valid.csv"
VALID_OUT = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_valid.csv"
TEST_OUT = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test_docs.csv"


def _set_field_size_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)


def _read_rows(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
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
    """
    Берём текущий VALID (документы) и делим его как будто это 30% всего корпуса:
    примерно 2/3 -> новый VALID (20% от полного), 1/3 -> новый TEST (10% от полного).

    VALID перезаписываем в тот же файл, TEST сохраняем в новый CSV.
    """
    _set_field_size_limit()

    rows, fieldnames = _read_rows(VALID_IN)
    total = len(rows)
    if total == 0:
        raise RuntimeError(f"Validation CSV is empty: {VALID_IN}")

    rnd = random.Random(42)
    rnd.shuffle(rows)

    # 20% : 10% ~= 2 : 1, то есть VALID:TEST ~= 2/3 : 1/3
    test_n = max(1, int(round(total / 3)))
    valid_n = max(1, total - test_n)

    # Пусть TEST будет первой частью, VALID — остальной
    test_rows = rows[:test_n]
    valid_rows = rows[test_n:]

    if len(valid_rows) != valid_n:
        # На всякий случай подправим, если из-за округления не совпало
        valid_rows = rows[test_n:test_n + valid_n]
        test_rows = rows[:test_n] + rows[test_n + valid_n :]

    _write_rows(VALID_OUT, valid_rows, fieldnames)
    _write_rows(TEST_OUT, test_rows, fieldnames)

    print(f"Total in original valid: {total}")
    print(f"New valid (≈20%): {len(valid_rows)} -> {VALID_OUT}")
    print(f"New test  (≈10%): {len(test_rows)} -> {TEST_OUT}")


if __name__ == "__main__":
    main()

