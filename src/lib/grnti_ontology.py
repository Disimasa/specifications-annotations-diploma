from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ontology_code_map(ontology_path: Path) -> Dict[str, str]:
    data = load_json(ontology_path)
    out: Dict[str, str] = {}
    for n in data.get("nodes", []):
        nid = n.get("id")
        code = n.get("code")
        if isinstance(nid, str) and isinstance(code, str) and code.strip():
            out[nid] = code.strip()
    return out


def load_ontology_texts(ontology_path: Path) -> Dict[str, str]:
    data = load_json(ontology_path)
    out: Dict[str, str] = {}
    for n in data.get("nodes", []):
        code = (n.get("code") or "").strip()
        if not code:
            continue
        full_label = (n.get("full_label") or n.get("label") or "").strip()
        llm_desc = (n.get("llm_description") or "").strip()
        parts: List[str] = []
        if full_label:
            parts.append(full_label)
        if llm_desc:
            parts.append(llm_desc)
        text = ". ".join(parts).strip(" .")
        if not text:
            text = full_label or code
        out[code] = text
    return out


def is_leaf_grnti_code(code: str) -> bool:
    parts = code.split(".")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


def to_level_code(code: str, level: int) -> Optional[str]:
    parts = code.split(".")
    if level <= 0 or level > len(parts):
        return None
    sub = parts[:level]
    if not all(p.isdigit() for p in sub):
        return None
    return ".".join(sub)


def aggregate_codes_to_level(codes: Sequence[str], level: int) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for c in codes:
        lc = to_level_code(c, level)
        if not lc or lc in seen:
            continue
        seen.add(lc)
        out.append(lc)
    return out
