from __future__ import annotations

"""
Нормализация ontology_grnti.json:
- выносим коды (типа "20.23.29") из label в отдельное поле code;
- чистим label до чистого названия ("Электронные библиотеки");
- добавляем full_label = "Родитель1. Родитель2. Текущий".

Результат пишем в data/ontology_grnti_clean.json, исходный файл не трогаем.
"""

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "data" / "ontology_grnti.json"
OUTPUT_PATH = PROJECT_DIR / "data" / "ontology_grnti_clean.json"

ROOT_ID = "http://example.org/grnti_root"


def extract_code_and_title(label: str) -> tuple[Optional[str], str]:
    """
    Извлекает код ГРНТИ из начала строки, если он там есть.
    Примеры:
    - "20 ИНФОРМАТИКА" -> ("20", "ИНФОРМАТИКА")
    - "20.23 Информационный поиск" -> ("20.23", "Информационный поиск")
    - "20.23.29 Электронные библиотеки" -> ("20.23.29", "Электронные библиотеки")
    - "ГРНТИ" -> (None, "ГРНТИ")
    """
    pattern = r"^(\d{2}(?:\.\d{2}){0,2})\s*(.*)$"
    m = re.match(pattern, label.strip())
    if not m:
        return None, label.strip()
    code = m.group(1)
    rest = (m.group(2) or "").strip(" :–-")
    title = rest or code
    return code, title


def build_parent_map(links: List[dict]) -> Dict[str, str]:
    """
    Строит parent_map[target] = source на основе links.
    Для ГРНТИ предполагаем, что у каждого узла не более одного родителя.
    """
    parent: Dict[str, str] = {}
    for link in links:
        source = link.get("source")
        target = link.get("target")
        if not source or not target:
            continue
        # Не трогаем корень, у него нет родителя
        if target == ROOT_ID:
            continue
        if target in parent and parent[target] != source:
            # В теории конфликт, но для ГРНТИ не ожидается
            # Оставляем первого родителя
            continue
        parent[target] = source
    return parent


def compute_full_label(node_id: str, nodes_by_id: Dict[str, dict], parent_map: Dict[str, str]) -> str:
    """
    Строит full_label в формате:
    "Раздел: {grandparent_label}. Область: {parent_label}. {label}"
    (без кодов, если они были в начале).

    - Для листовых узлов ГРНТИ (3-й уровень) получаем контекст из двух предков.
    - Для 2-го уровня будет только "Область: {parent_label}. {label}" или просто "{label}".
    - Для верхнего уровня и корня оставляем только собственный label.
    """
    if node_id == ROOT_ID:
        return nodes_by_id[node_id]["label"]

    node = nodes_by_id.get(node_id)
    if not node:
        return ""

    label = (node.get("label") or "").strip()

    # Находим родителя и "деда" по parent_map
    parent_id = parent_map.get(node_id)
    parent_label = None
    grandparent_label = None

    if parent_id and parent_id != ROOT_ID:
        parent_node = nodes_by_id.get(parent_id)
        if parent_node:
            parent_label = (parent_node.get("label") or "").strip()

        grand_id = parent_map.get(parent_id)
        if grand_id and grand_id != ROOT_ID:
            grand_node = nodes_by_id.get(grand_id)
            if grand_node:
                grandparent_label = (grand_node.get("label") or "").strip()

    parts: List[str] = []
    if grandparent_label:
        parts.append(f"Раздел: {grandparent_label}")
    if parent_label:
        parts.append(f"Область: {parent_label}")
    if label:
        parts.append(label)

    return ". ".join(parts)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Не найден входной файл {INPUT_PATH}")

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    nodes: List[dict] = data.get("nodes", [])
    links: List[dict] = data.get("links", [])

    # Индекс по id
    nodes_by_id: Dict[str, dict] = {n["id"]: n for n in nodes if "id" in n}

    # 1) выделяем code + чистим label/description
    for node in nodes:
        label = node.get("label", "")
        code, title = extract_code_and_title(label)
        if code is not None:
            node["code"] = code
        # Переписываем label на "чистый" текст без кода
        node["label"] = title

        # Описание можно тоже сделать более чистым: если оно было "код + текст", чистим аналогично
        desc = node.get("description")
        if isinstance(desc, str) and desc.strip():
            _, clean_desc = extract_code_and_title(desc)
            node["description"] = clean_desc

    # 2) строим parent_map по links
    parent_map = build_parent_map(links)

    # 3) считаем full_label для всех узлов
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        full = compute_full_label(node_id, nodes_by_id, parent_map)
        if full:
            node["full_label"] = full

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Нормализованная онтология сохранена в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

