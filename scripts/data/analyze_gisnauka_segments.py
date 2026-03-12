from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import fmean, median
from typing import Dict, List


PROJECT_DIR = Path(__file__).resolve().parents[2]
SEGMENTS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_train.csv"


def _stats(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {}
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    def pct(p: float) -> float:
        if n == 1:
            return arr_sorted[0]
        idx = int(p * (n - 1))
        return float(arr_sorted[idx])
    return {
        "min": float(arr_sorted[0]),
        "max": float(arr_sorted[-1]),
        "mean": float(fmean(arr_sorted)),
        "median": float(median(arr_sorted)),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
    }


def main() -> None:
    if not SEGMENTS_CSV.exists():
        raise FileNotFoundError(f"Не найден файл сегментов: {SEGMENTS_CSV}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    lengths: List[int] = []
    word_counts: List[int] = []
    by_doc: Dict[str, int] = {}

    with SEGMENTS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = (row.get("doc_id") or "").strip()
            text = (row.get("segment_text") or "").strip()
            if not text:
                continue
            l = len(text)
            lengths.append(l)
            wc = len(text.split())
            word_counts.append(wc)
            by_doc[doc_id] = by_doc.get(doc_id, 0) + 1

    seg_count = len(lengths)
    doc_count = len(by_doc)

    seg_len_stats = _stats(lengths)
    seg_wc_stats = _stats(word_counts)
    segs_per_doc_stats = _stats(list(by_doc.values()))

    print("segments_count:", seg_count)
    print("docs_count:", doc_count)
    print("segment_length_chars:", seg_len_stats)
    print("segment_length_words:", seg_wc_stats)
    print("segments_per_doc:", segs_per_doc_stats)


if __name__ == "__main__":
    main()

