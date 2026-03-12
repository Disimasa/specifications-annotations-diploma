from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
SEGMENTS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_train.csv"


def main() -> None:
    if not SEGMENTS_CSV.exists():
        raise FileNotFoundError(f"Не найден файл сегментов: {SEGMENTS_CSV}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    max_row = None
    max_len = 0

    with SEGMENTS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("segment_text") or "").strip()
            l = len(text)
            if l > max_len:
                max_len = l
                max_row = row

    if not max_row:
        print("В файле нет непустых сегментов.")
        return

    text = (max_row.get("segment_text") or "").strip()
    print(f"len_chars: {len(text)}")
    print(f"doc_id: {max_row.get('doc_id')}")
    print(f"segment_index: {max_row.get('segment_index')}")
    print("--- SEGMENT START ---")
    print(text)
    print("--- SEGMENT END ---")


if __name__ == "__main__":
    main()

