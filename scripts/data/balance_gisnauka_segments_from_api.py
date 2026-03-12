from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import requests

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from annotation.segmenter import TextSegmenter
from annotation.segment_filter import SegmentFilter


ONTOLOGY_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
FILTERED_SEGMENTS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_train_filtered.csv"
OUT_SEGMENTS_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_segments_train_balanced.csv"

GISNAUKA_API_URL = "https://gisnauka.ru/api/egisu/base/search"
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 1.5
MAX_PAGES_PER_QUERY = 50
PAGE_SIZE = 10

SESSION_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin": "https://gisnauka.ru",
    "Referer": "https://gisnauka.ru/global-search",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}


def is_leaf_grnti_code(code: str) -> bool:
    parts = code.split(".")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


def load_filtered_counts() -> Counter[str]:
    if not FILTERED_SEGMENTS_CSV.exists():
        raise FileNotFoundError(f"Не найден отфильтрованный сегментный датасет: {FILTERED_SEGMENTS_CSV}")
    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)
    counts: Counter[str] = Counter()
    with FILTERED_SEGMENTS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            codes_raw = (row.get("grnti_codes") or "").strip()
            if not codes_raw:
                continue
            codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
            for code in codes:
                if is_leaf_grnti_code(code):
                    counts[code] += 1
    return counts


def load_top_levels() -> List[tuple[str, str]]:
    if not ONTOLOGY_PATH.exists():
        raise FileNotFoundError(f"Онтология не найдена: {ONTOLOGY_PATH}")
    data = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    out: List[tuple[str, str]] = []
    seen: set[str] = set()
    for n in nodes:
        code = (n.get("code") or "").strip()
        label = (n.get("label") or "").strip()
        if not code or not label:
            continue
        if len(code) == 2 and code.isdigit() and code not in seen:
            seen.add(code)
            out.append((code, label))
    return sorted(out, key=lambda x: x[0])


def search_gisnauka(query: str, page: int) -> List[Dict[str, Any]]:
    payload = {
        "search_query": query,
        "critical_technologies": [],
        "dissertations": True,
        "full_text_available": False,
        "ikrbses": True,
        "nioktrs": True,
        "organization": [],
        "page": page,
        "priority_directions": [],
        "rids": True,
        "rubrics": [],
        "search_area": "Во всех полях",
        "sort_by": "Дата регистрации",
        "open_license": False,
        "free_licenses": False,
        "expert_estimation_exist": False,
    }
    resp = requests.post(
        GISNAUKA_API_URL,
        headers=SESSION_HEADERS,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        return []
    hits = data.get("hits", {})
    if isinstance(hits, dict):
        inner = hits.get("hits")
        if isinstance(inner, list):
            return inner
    return []


def extract_codes_from_item(hit: Dict[str, Any]) -> List[str]:
    codes: List[str] = []
    src = hit.get("_source", {})
    rubrics = src.get("rubrics") or []
    if isinstance(rubrics, list):
        for r in rubrics:
            if not isinstance(r, dict):
                continue
            code = r.get("code")
            if isinstance(code, str):
                code = code.strip()
                if is_leaf_grnti_code(code) and code not in codes:
                    codes.append(code)
    return codes


def extract_title(hit: Dict[str, Any]) -> str:
    src = hit.get("_source", {})
    title = src.get("name") or ""
    return str(title).strip()


def extract_abstract(hit: Dict[str, Any]) -> str:
    src = hit.get("_source", {})
    abstract = src.get("abstract") or src.get("nioktr", {}).get("annotation") or ""
    return str(abstract).strip()


def main() -> None:
    base_counts = load_filtered_counts()
    if not base_counts:
        print("В отфильтрованном датасете нет листовых классов.")
        return

    target = max(base_counts.values())
    needed: Dict[str, int] = {code: target - cnt for code, cnt in base_counts.items() if cnt < target}

    print(f"Классов с дефицитом: {len(needed)}")
    print(f"Максимум сегментов для класса (target): {target}")

    OUT_SEGMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Сначала копируем исходный отфильтрованный датасет
    with FILTERED_SEGMENTS_CSV.open(encoding="utf-8", newline="") as f_in, OUT_SEGMENTS_CSV.open(
        encoding="utf-8", newline="", mode="w"
    ) as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or [
            "doc_id",
            "segment_index",
            "segment_text",
            "top_code",
            "top_label",
            "grnti_codes",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        base_rows = 0
        for row in reader:
            writer.writerow(row)
            base_rows += 1

    print(f"Скопировано исходных сегментов: {base_rows}")

    segmenter = TextSegmenter()
    seg_filter = SegmentFilter()
    added_segments = 0

    top_levels = load_top_levels()
    seen_docs: set[str] = set()

    for top_code, top_label in top_levels:
        if all(not code.startswith(top_code + ".") for code in needed.keys()):
            continue
        print(f"== Раздел {top_code} :: {top_label} ==")

        page = 1
        while page <= MAX_PAGES_PER_QUERY and any(v > 0 for v in needed.values()):
            try:
                hits = search_gisnauka(top_label, page)
            except Exception as e:
                print(f"  Ошибка запроса страницы {page}: {e}")
                break
            if not hits:
                break

            for hit in hits:
                src = hit.get("_source", {})
                doc_id = str(src.get("_id") or src.get("id") or "").strip()
                if not doc_id:
                    # fallback к hash title+abstract
                    doc_id = f"hit_{top_code}_{page}_{len(seen_docs)}"
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)

                title = extract_title(hit)
                abstract = extract_abstract(hit)
                if not title and not abstract:
                    continue
                text = f"{title}\n\n{abstract}".strip()
                if not text:
                    continue

                codes = extract_codes_from_item(hit)
                codes = [c for c in codes if c in needed and needed[c] > 0]
                if not codes:
                    continue

                segments = segmenter.segment(text)
                segments = seg_filter.filter_segments(segments)
                segments = [s.strip() for s in segments if s and s.strip()]
                if not segments:
                    continue
                if len(segments) >= 14:
                    continue

                valid_segments = [s for s in segments if 60 <= len(s) <= 600]
                if not valid_segments:
                    continue

                with OUT_SEGMENTS_CSV.open(encoding="utf-8", newline="", mode="a") as f_out:
                    writer = csv.DictWriter(
                        f_out,
                        fieldnames=[
                            "doc_id",
                            "segment_index",
                            "segment_text",
                            "top_code",
                            "top_label",
                            "grnti_codes",
                        ],
                    )
                    for idx, seg in enumerate(valid_segments):
                        any_used = False
                        for code in list(codes):
                            if needed.get(code, 0) <= 0:
                                continue
                            writer.writerow(
                                {
                                    "doc_id": doc_id,
                                    "segment_index": idx,
                                    "segment_text": seg,
                                    "top_code": top_code,
                                    "top_label": top_label,
                                    "grnti_codes": ";".join(sorted(set(codes))),
                                }
                            )
                            needed[code] -= 1
                            added_segments += 1
                            any_used = True
                            if needed[code] <= 0:
                                del needed[code]
                            if not needed:
                                break
                        if not needed:
                            break

                if not needed:
                    break

            if not needed:
                break

            page += 1
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        if not needed:
            print("Все дефициты по классам заполнены.")
            break

    remaining = sum(v for v in needed.values() if v > 0)
    print(f"Добавлено новых сегментов: {added_segments}")
    print(f"Осталось неудовлетворённых запросов сегментов: {remaining}")


if __name__ == "__main__":
    main()

