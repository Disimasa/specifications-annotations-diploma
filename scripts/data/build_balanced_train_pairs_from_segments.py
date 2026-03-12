from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[2]
BALANCED_SEGMENTS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_train_balanced.csv"
OUTPUT_PAIRS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_train_pairs_balanced_bs16.csv"


def is_leaf_grnti_code(code: str) -> bool:
    parts = code.split(".")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


def load_candidates() -> List[Tuple[str, int, str, str, str]]:
    if not BALANCED_SEGMENTS_CSV.exists():
        raise FileNotFoundError(f"Не найден файл сегментов: {BALANCED_SEGMENTS_CSV}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    candidates: List[Tuple[str, int, str, str, str]] = []
    with BALANCED_SEGMENTS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = (row.get("doc_id") or "").strip()
            if not doc_id:
                continue
            segment_index_raw = (row.get("segment_index") or "").strip()
            try:
                segment_index = int(segment_index_raw)
            except ValueError:
                segment_index = 0
            segment_text = (row.get("segment_text") or "").strip()
            if not segment_text:
                continue
            top_code = (row.get("top_code") or "").strip()
            if not top_code:
                continue
            codes_raw = (row.get("grnti_codes") or "").strip()
            if not codes_raw:
                continue
            codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
            codes = [c for c in codes if is_leaf_grnti_code(c)]
            if not codes:
                continue
            for code in codes:
                top_section = code.split(".")[0] if "." in code else code[:2]
                candidates.append((doc_id, segment_index, segment_text, code, top_section))
    return candidates


def build_balanced_pairs(
    candidates: List[Tuple[str, int, str, str, str]],
    target_per_top: int | None = None,
) -> Tuple[List[int], Dict[str, int]]:
    import random

    top_to_code_to_indices: Dict[str, Dict[str, List[int]]] = {}
    for idx, (_doc_id, _seg_idx, _text, code, top_section) in enumerate(candidates):
        code_map = top_to_code_to_indices.setdefault(top_section, {})
        lst = code_map.setdefault(code, [])
        lst.append(idx)

    if not top_to_code_to_indices:
        return [], {}

    top_sections = sorted(top_to_code_to_indices.keys())
    uniq_counts: Dict[str, int] = {top: len(top_to_code_to_indices[top]) for top in top_sections}
    if not uniq_counts:
        return [], {}

    if target_per_top is not None and target_per_top > 0:
        k_global = int(target_per_top)
    else:
        k_global = min(uniq_counts.values())
        if k_global <= 0:
            return [], uniq_counts

    rnd = random.Random(42)
    selected: List[int] = []

    for top_section in top_sections:
        code_map = top_to_code_to_indices[top_section]
        codes = list(code_map.keys())
        if not codes:
            continue
        rnd.shuffle(codes)

        k = min(k_global, len(codes))
        chosen_codes = codes[:k]
        for code in chosen_codes:
            idx_list = code_map.get(code)
            if not idx_list:
                continue
            idx = idx_list[0]
            selected.append(idx)

    return selected, uniq_counts


def save_pairs(
    candidates: List[Tuple[str, int, str, str, str]],
    indices: List[int],
) -> None:
    OUTPUT_PAIRS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PAIRS_CSV.open(encoding="utf-8", newline="", mode="w") as f:
        fieldnames = ["doc_id", "segment_index", "segment_text", "leaf_code", "top_section"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in indices:
            doc_id, seg_idx, text, code, top_section = candidates[idx]
            writer.writerow(
                {
                    "doc_id": doc_id,
                    "segment_index": seg_idx,
                    "segment_text": text,
                    "leaf_code": code,
                    "top_section": top_section,
                }
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-per-top",
        type=int,
        default=0,
        help="Желаемое количество уникальных листовых специализаций на каждый верхний раздел. "
        "Если 0, берётся минимально возможное по данным.",
    )
    args = parser.parse_args()

    candidates = load_candidates()
    if not candidates:
        print("Нет кандидатов для построения выборки.")
        return

    target = int(args.target_per_top) if int(args.target_per_top) > 0 else None
    selected_indices, uniq_counts = build_balanced_pairs(candidates, target_per_top=target)

    print("Уникальные листовые коды по верхним разделам (в исходных данных):")
    for top_section in sorted(uniq_counts.keys()):
        print(f"{top_section}: {uniq_counts[top_section]}")

    if not selected_indices:
        print("Не удалось построить выборку с заданными ограничениями.")
        return

    save_pairs(candidates, selected_indices)

    top_counter: Counter[str] = Counter()
    leaf_counter: Counter[str] = Counter()
    for idx in selected_indices:
        _doc_id, _seg_idx, _text, code, top_section = candidates[idx]
        top_counter[top_section] += 1
        leaf_counter[code] += 1

    total = len(selected_indices)
    print(f"Итоговый размер выборки: {total}")
    print("Распределение по разделам верхнего уровня:")
    for top_section, cnt in sorted(top_counter.items(), key=lambda x: x[0]):
        frac = cnt / float(total) if total else 0.0
        print(f"{top_section}: {cnt} ({frac:.4f})")

    non_unique_repeats = sum(cnt - 1 for cnt in leaf_counter.values() if cnt > 1)
    print(f"Повторений неуникальных листовых кодов (сверх одного раза): {non_unique_repeats}")


if __name__ == "__main__":
    main()

