from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
SEGMENTS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_train.csv"
OUT_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_train_filtered.csv"


def is_leaf_grnti_code(code: str) -> bool:
    parts = code.split(".")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


def main() -> None:
    if not SEGMENTS_CSV.exists():
        raise FileNotFoundError(f"Не найден файл сегментов: {SEGMENTS_CSV}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    segments_per_doc: dict[str, int] = {}
    with SEGMENTS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = (row.get("doc_id") or "").strip()
            if not doc_id:
                continue
            segments_per_doc[doc_id] = segments_per_doc.get(doc_id, 0) + 1

    kept_doc_ids = {doc_id for doc_id, cnt in segments_per_doc.items() if cnt < 14}

    with SEGMENTS_CSV.open(encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or [
            "doc_id",
            "segment_index",
            "segment_text",
            "top_code",
            "top_label",
            "grnti_codes",
        ]

        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUT_CSV.open(encoding="utf-8", newline="", mode="w") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            total_segments = 0
            kept_segments = 0
            top_section_counter: Counter[str] = Counter()
            leaf_code_counter: Counter[str] = Counter()

            for row in reader:
                total_segments += 1
                doc_id = (row.get("doc_id") or "").strip()
                if doc_id not in kept_doc_ids:
                    continue
                text = (row.get("segment_text") or "").strip()
                l = len(text)
                if l < 60 or l > 600:
                    continue
                writer.writerow(row)
                kept_segments += 1

                codes_raw = (row.get("grnti_codes") or "").strip()
                if codes_raw:
                    codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
                    for code in codes:
                        if not is_leaf_grnti_code(code):
                            continue
                        leaf_code_counter[code] += 1
                        top_section = code.split(".")[0]
                        top_section_counter[top_section] += 1

    print(f"Всего сегментов в исходном файле: {total_segments}")
    print(f"Сегментов после фильтрации: {kept_segments}")
    print(f"Документов с количеством сегментов < 14: {len(kept_doc_ids)}")

    print("Покрытие верхнеуровневых разделов (XX):")
    for top, cnt in sorted(top_section_counter.items(), key=lambda x: x[0]):
        print(f"  {top}: {cnt}")

    print("Всего листовых специализаций в выборке:", len(leaf_code_counter))


if __name__ == "__main__":
    main()

