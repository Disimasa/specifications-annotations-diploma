from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in map(str, (__import__("sys").path)):
    import sys as _sys

    _sys.path.insert(0, str(SRC_DIR))

from annotation.ontology import Ontology  # noqa: E402
from lib.ontology_embeddings_registry import (  # noqa: E402
    DEFAULT_EMBEDDINGS_PATH,
    _safe_name,
)


DEFAULT_ONTOLOGY = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"


def _build_texts_for_ontology(ontology_path: Path) -> Tuple[Ontology, Dict[str, dict], List[str], List[str]]:
    with ontology_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ontology = Ontology(data)
    nodes = data["nodes"]
    id_to_node: Dict[str, dict] = {str(n["id"]): n for n in nodes}

    texts: List[str] = []
    comp_ids: List[str] = []
    for comp in ontology.competencies:
        node = id_to_node.get(comp.id)
        if node is None:
            continue
        llm_desc = (node.get("llm_description") or "").strip()
        full_label = (node.get("full_label") or "").strip()
        parts: List[str] = []
        if full_label:
            parts.append(full_label)
        if llm_desc:
            parts.append(llm_desc)
        text = ". ".join(parts).strip(". ").strip()
        if not text:
            continue
        texts.append(text)
        comp_ids.append(comp.id)
    return ontology, id_to_node, comp_ids, texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Предварительно посчитать эмбеддинги онтологии для модели.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Имя / путь модели SentenceTransformer (например deepvk/USER-bge-m3 или путь к best‑папке).",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=DEFAULT_ONTOLOGY,
        help="Путь к JSON‑онтологии (по умолчанию data/ontology_grnti_with_llm.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Путь для NPZ с эмбеддингами. "
            "По умолчанию: data/ontology_grnti_embeddings_<метка_модели>.npz "
            "(метка = basename модели, с безопасными символами)."
        ),
    )
    args = parser.parse_args()

    model_name = args.model
    ontology_path = args.ontology

    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology not found: {ontology_path}")

    if args.out is not None:
        out_path = args.out
    else:
        safe = _safe_name(model_name)
        out_path = DEFAULT_EMBEDDINGS_PATH.with_name(f"ontology_grnti_embeddings_{safe}.npz")

    print(f"Модель: {model_name}")
    print(f"Онтология: {ontology_path}")
    print(f"Выходной файл NPZ: {out_path}")

    _, _, comp_ids, texts = _build_texts_for_ontology(ontology_path)
    if not texts:
        raise RuntimeError("В онтологии не нашлось ни одной компетенции с текстом для кодирования.")

    model = SentenceTransformer(model_name)
    embs = model.encode(
        texts,
        convert_to_tensor=False,
        normalize_embeddings=True,
        batch_size=64,
    )

    embs_arr = np.asarray(embs, dtype="float32")
    ids_arr = np.asarray(comp_ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, ids=ids_arr, embeddings=embs_arr)

    print(f"Сохранено {len(comp_ids)} эмбеддингов в {out_path}")


if __name__ == "__main__":
    main()

