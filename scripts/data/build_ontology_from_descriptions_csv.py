from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
DEFAULT_CSV = PROJECT_DIR / "data" / "ontology_descriptions" / "grnti_descriptions_yagpt.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "ontology_grnti_with_yagpt.json"


def _load_descriptions_csv(path: Path) -> Dict[str, Dict[str, str]]:
    """Индекс по node_id: llm_description, llm_model, code."""
    by_id: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            node_id = (row.get("node_id") or "").strip()
            llm_desc = (row.get("llm_description") or "").strip()
            if not node_id or not llm_desc:
                continue
            by_id[node_id] = {
                "llm_description": llm_desc,
                "llm_model": (row.get("llm_model") or "").strip(),
                "code": (row.get("code") or "").strip(),
            }
    return by_id


def build_ontology(
    ontology_path: Path,
    descriptions_csv: Path,
    output_path: Path,
    *,
    clear_old_llm: bool = True,
) -> Dict[str, Any]:
    data = json.loads(ontology_path.read_text(encoding="utf-8"))
    nodes: List[dict] = data.get("nodes", [])
    desc_by_id = _load_descriptions_csv(descriptions_csv)

    updated = 0
    cleared = 0
    missing_in_csv: List[str] = []

    for node in nodes:
        node_id = str(node.get("id") or "")
        if clear_old_llm and "llm_description" in node:
            node.pop("llm_description", None)
            node.pop("llm_model", None)
            cleared += 1

        entry = desc_by_id.get(node_id)
        if entry is None:
            code = (node.get("code") or "").strip()
            if code and code.count(".") >= 2:
                missing_in_csv.append(code)
            continue

        node["llm_description"] = entry["llm_description"]
        if entry["llm_model"]:
            node["llm_model"] = entry["llm_model"]
        updated += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "ontology_nodes": len(nodes),
        "csv_descriptions": len(desc_by_id),
        "nodes_with_llm_description": updated,
        "leaf_codes_without_csv": len(missing_in_csv),
        "output_path": str(output_path),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Собрать JSON онтологии с llm_description из CSV (YaGPT/GigaChat и т.п.)."
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=DEFAULT_ONTOLOGY,
        help=f"Базовая онтология (по умолчанию {DEFAULT_ONTOLOGY.name}).",
    )
    parser.add_argument(
        "--descriptions-csv",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV с колонками node_id, llm_description, llm_model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Выходной JSON (по умолчанию {DEFAULT_OUTPUT.name}).",
    )
    parser.add_argument(
        "--keep-old-llm",
        action="store_true",
        help="Не удалять старые llm_description у узлов без записи в CSV.",
    )
    args = parser.parse_args()

    if not args.ontology.is_file():
        raise FileNotFoundError(f"Онтология не найдена: {args.ontology}")
    if not args.descriptions_csv.is_file():
        raise FileNotFoundError(f"CSV не найден: {args.descriptions_csv}")

    stats = build_ontology(
        args.ontology,
        args.descriptions_csv,
        args.output,
        clear_old_llm=not args.keep_old_llm,
    )
    print(f"Сохранено: {stats['output_path']}")
    print(f"  узлов в онтологии: {stats['ontology_nodes']}")
    print(f"  описаний в CSV: {stats['csv_descriptions']}")
    print(f"  узлов с llm_description: {stats['nodes_with_llm_description']}")
    if stats["leaf_codes_without_csv"]:
        print(f"  листьев без описания в CSV: {stats['leaf_codes_without_csv']}")


if __name__ == "__main__":
    main()
