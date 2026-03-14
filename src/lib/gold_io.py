from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .grnti_ontology import is_leaf_grnti_code


@dataclass
class GoldItem:
    doc_id: str
    gold_codes: Tuple[str, ...]
    text: Optional[str] = None
    text_path: Optional[Path] = None
    top_code: Optional[str] = None


def _csv_field_size() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)


def read_gold_csv(path: Path) -> List[GoldItem]:
    items: List[GoldItem] = []
    _csv_field_size()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            codes_raw = (row.get("grnti_codes") or "").strip()
            if not codes_raw:
                continue
            codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
            codes = [c for c in codes if is_leaf_grnti_code(c)]
            codes = sorted(set(codes))
            if not codes:
                continue
            text = f"{title}\n\n{abstract}".strip()
            if not text:
                continue
            doc_id = (row.get("doc_id") or "").strip() or f"gisnauka_{i}"
            top_code = (row.get("top_code") or "").strip() or codes[0].split(".")[0]
            items.append(
                GoldItem(
                    doc_id=doc_id,
                    gold_codes=tuple(codes),
                    text=text,
                    top_code=top_code,
                )
            )
    return items


def read_gold_jsonl(
    path: Path,
    default_texts_dir: Optional[Path] = None,
) -> List[GoldItem]:
    if default_texts_dir is None:
        default_texts_dir = path.parent.parent / "specifications" / "texts"
    items: List[GoldItem] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        doc_id = str(obj.get("doc_id", "")).strip()
        if not doc_id:
            continue
        gold_codes_raw = obj.get("gold_codes") or obj.get("labels") or []
        gold_codes: List[str] = []
        if isinstance(gold_codes_raw, list):
            for x in gold_codes_raw:
                if isinstance(x, str):
                    gold_codes.append(x.strip())
                elif isinstance(x, dict) and isinstance(x.get("code"), str):
                    gold_codes.append(x["code"].strip())
        gold_codes = [c for c in gold_codes if is_leaf_grnti_code(c)]
        gold_codes = sorted(set(gold_codes))
        if not gold_codes:
            continue
        tp = obj.get("text_path")
        if isinstance(tp, str) and tp.strip():
            text_path = Path(tp)
        else:
            text_path = default_texts_dir / f"{doc_id}.txt"
        items.append(
            GoldItem(doc_id=doc_id, gold_codes=tuple(gold_codes), text_path=text_path)
        )
    return items


def read_gold_items(
    gold_path: Path,
    default_texts_dir: Optional[Path] = None,
) -> List[GoldItem]:
    if gold_path.suffix.lower() == ".csv":
        return read_gold_csv(gold_path)
    return read_gold_jsonl(gold_path, default_texts_dir=default_texts_dir)


def read_valid_segments(path: Path) -> List[Tuple[str, str, List[str]]]:
    rows: List[Tuple[str, str, List[str]]] = []
    _csv_field_size()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            doc_id = (row.get("doc_id") or "").strip() or f"valid_{i}"
            segment_text = (row.get("segment_text") or "").strip()
            if not segment_text:
                continue
            codes_raw = (row.get("grnti_codes") or "").strip()
            if not codes_raw:
                continue
            codes = [c.strip() for c in codes_raw.split(";") if c.strip()]
            codes = [c for c in codes if is_leaf_grnti_code(c)]
            if not codes:
                continue
            rows.append((doc_id, segment_text, codes))
    return rows
