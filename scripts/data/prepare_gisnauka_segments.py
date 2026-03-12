from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from annotation.segmenter import TextSegmenter
from annotation.segment_filter import SegmentFilter


SRC_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_train.csv"
OUT_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_train.csv"


def is_leaf_grnti_code(code: str) -> bool:
    parts = code.split(".")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


def main() -> None:
    if not SRC_CSV.exists():
        raise FileNotFoundError(f"Источник не найден: {SRC_CSV}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    segmenter = TextSegmenter()
    segment_filter = SegmentFilter()

    with SRC_CSV.open(encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        src_fields = reader.fieldnames or []

    out_fields: List[str] = [
        "doc_id",
        "segment_index",
        "segment_text",
        "top_code",
        "top_label",
        "grnti_codes",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with OUT_CSV.open(encoding="utf-8", newline="", mode="w") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for i, row in enumerate(rows):
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            text = f"{title}\n\n{abstract}".strip()
            if not text:
                continue

            codes_raw = (row.get("grnti_codes") or "").strip()
            if not codes_raw:
                continue
            codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
            codes = [c for c in codes if is_leaf_grnti_code(c)]
            if not codes:
                continue

            doc_id = (row.get("doc_id") or "").strip() or f"gisnauka_{i}"
            top_code = (row.get("top_code") or "").strip()
            top_label = (row.get("top_label") or "").strip()

            segments = segmenter.segment(text)
            segments = segment_filter.filter_segments(segments)
            segments = [s.strip() for s in segments if s and s.strip()]
            if not segments:
                continue

            for idx, seg in enumerate(segments):
                writer.writerow(
                    {
                        "doc_id": doc_id,
                        "segment_index": idx,
                        "segment_text": seg,
                        "top_code": top_code,
                        "top_label": top_label,
                        "grnti_codes": ";".join(codes),
                    }
                )
                written += 1

    print(f"Сегментированный train сохранён в {OUT_CSV}, строк: {written}")


if __name__ == "__main__":
    main()

