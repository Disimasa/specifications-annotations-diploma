from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from annotation.segment_filter import SegmentFilter
from annotation.segmenter import TextSegmenter


TRAIN_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_train.csv"
TEST_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test.csv"
ONTOLOGY_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
OUT_CSV = PROJECT_DIR / "data" / "gold" / "gisnauka_samples_train_augmented.csv"

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


def load_existing_samples() -> Tuple[Counter[str], set[Tuple[str, str]]]:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"train CSV not found: {TRAIN_CSV}")

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    counts: Counter[str] = Counter()
    seen_keys: set[Tuple[str, str]] = set()

    def read_file(path: Path) -> None:
        if not path.exists():
            return
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = (row.get("title") or "").strip()
                abstract = (row.get("abstract") or "").strip()
                if title or abstract:
                    key = (title.lower(), abstract.lower())
                    seen_keys.add(key)
                codes_raw = (row.get("grnti_codes") or "").strip()
                if not codes_raw:
                    continue
                codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
                for code in codes:
                    if is_leaf_grnti_code(code):
                        counts[code] += 1

    read_file(TRAIN_CSV)
    read_file(TEST_CSV)
    return counts, seen_keys


def load_top_labels() -> Dict[str, Tuple[str, str]]:
    if not ONTOLOGY_PATH.exists():
        raise FileNotFoundError(f"Онтология не найдена: {ONTOLOGY_PATH}")
    data = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    by_code: Dict[str, Tuple[str, str]] = {}
    for n in nodes:
        code = (n.get("code") or "").strip()
        if not code:
            continue
        top_code = code.split(".")[0]
        label = (n.get("label") or "").strip()
        full_label = (n.get("full_label") or "").strip() or label
        by_code[code] = (top_code, full_label)
    return by_code


def make_rubric_id(code: str) -> str:
    digits = code.replace(".", "").strip()
    return f"-{digits}10000000000000001"


def search_gisnauka_by_rubric(rubric_id: str, page: int) -> List[Dict[str, Any]]:
    payload = {
        "search_query": None,
        "critical_technologies": [],
        "dissertations": True,
        "full_text_available": False,
        "ikrbses": True,
        "nioktrs": True,
        "organization": [],
        "page": page,
        "priority_directions": [],
        "rids": True,
        "rubrics": [rubric_id],
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


def extract_codes(hit: Dict[str, Any]) -> List[str]:
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
    counts, seen_keys = load_existing_samples()
    if not counts:
        print("Нет исходных сэмплов в train/test для подсчёта частот.")
        return

    target = max(counts.values())
    print(f"Целевое количество сэмплов на листовую специализацию: {target}")

    code_to_top_label = load_top_labels()

    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["top_code", "top_label", "title", "abstract", "grnti_codes"]

    with TRAIN_CSV.open(encoding="utf-8", newline="") as f_in, OUT_CSV.open(
        encoding="utf-8",
        newline="",
        mode="w",
    ) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        base_rows = 0
        for row in reader:
            writer.writerow(row)
            base_rows += 1

    print(f"Скопировано исходных train-сэмплов: {base_rows}")

    segmenter = TextSegmenter()
    seg_filter = SegmentFilter()

    added = 0

    for leaf_code, current_count in sorted(counts.items(), key=lambda x: x[0]):
        need = target - current_count
        if need <= 0:
            continue
        top_code, top_label = code_to_top_label.get(leaf_code, (leaf_code.split(".")[0], leaf_code))
        rubric_id = make_rubric_id(leaf_code)
        print(f"== Листовой код {leaf_code} (top {top_code}) нужно добавить {need} сэмплов, рубрика {rubric_id}")

        page = 1
        while need > 0 and page <= MAX_PAGES_PER_QUERY:
            try:
                hits = search_gisnauka_by_rubric(rubric_id, page)
            except Exception as e:
                print(f"  Ошибка запроса страницы {page} для {leaf_code}: {e}")
                break
            if not hits:
                break

            with OUT_CSV.open(encoding="utf-8", newline="", mode="a") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                for hit in hits:
                    title = extract_title(hit)
                    abstract = extract_abstract(hit)
                    if not title and not abstract:
                        continue
                    key = (title.lower(), abstract.lower())
                    if key in seen_keys:
                        continue
                    text = f"{title}\n\n{abstract}".strip()
                    if not text:
                        continue
                    codes = extract_codes(hit)
                    codes = [c for c in codes if is_leaf_grnti_code(c)]
                    if leaf_code not in codes:
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
                    writer.writerow(
                        {
                            "top_code": top_code,
                            "top_label": top_label,
                            "title": title,
                            "abstract": abstract,
                            "grnti_codes": ";".join(sorted(set(codes))),
                        }
                    )
                    seen_keys.add(key)
                    added += 1
                    need -= 1
                    if need <= 0:
                        break

            if need <= 0:
                break
            page += 1
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        if need > 0:
            print(f"  Не удалось добрать {need} сэмплов для {leaf_code}")

    print(f"Готово. Всего добавлено новых train-сэмплов: {added}")
    print(f"Файл с дополненным train: {OUT_CSV}")


if __name__ == "__main__":
    main()

