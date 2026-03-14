from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
OUTPUT_DIR = PROJECT_DIR / "data"

COMP_PREFIX = "http://example.org/competencies#"


DEFAULT_MODEL_NAME = "deepvk/USER-bge-m3"


def build_text_for_embedding(node: dict) -> str:
    parts: List[str] = []
    llm_desc = (node.get("llm_description") or "").strip()
    full_label = (node.get("full_label") or "").strip()

    if full_label:
        parts.append(full_label)
    if llm_desc:
        parts.append(llm_desc)

    return ". ".join(parts).strip(". ").strip()


def _normalize_tag(raw: str) -> str:
    tag = raw.strip()
    if not tag:
        tag = "default"
    return tag.replace(" ", "_").replace("\\", "_").replace("/", "_")


def _make_output_path(model_name: str, tag_override: str | None = None) -> Path:
    if tag_override:
        tag = _normalize_tag(tag_override)
    else:
        candidate = model_name.strip()
        if "/" in candidate:
            tag = candidate.split("/")[-1]
        else:
            tag = Path(candidate).name or candidate
        tag = _normalize_tag(tag)
    return OUTPUT_DIR / f"ontology_grnti_embeddings_{tag}.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Путь к модели bi-encoder (локальная папка или HF id)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Полный путь к выходному .npz. Если не задан, путь формируется автоматически.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help=(
            "Метка для имени файла эмбеддингов (ontology_grnti_embeddings_<tag>.npz). "
            "Удобно задать таймстемп модели из best‑папки. "
            "Если не задана, используется basename модели."
        ),
    )
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Не найден входной файл {INPUT_PATH}")

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])

    comp_nodes = [n for n in nodes if str(n.get("id", "")).startswith(COMP_PREFIX)]

    ids: List[str] = []
    texts: List[str] = []

    for node in comp_nodes:
        node_id = node["id"]
        text = build_text_for_embedding(node)
        if not text:
            continue
        ids.append(node_id)
        texts.append(text)

    if not ids:
        raise RuntimeError("Не найдено ни одной компетенции с непустым текстом для эмбеддинга.")

    print(f"Всего компетенций для эмбеддингов: {len(ids)}")
    print(f"Модель для эмбеддингов: {args.model}")

    model = SentenceTransformer(args.model)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype(np.float32)

    if args.output:
        output_path = Path(args.output)
    else:
        tag_override = args.tag or None
        output_path = _make_output_path(args.model, tag_override=tag_override)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.array(ids, dtype=object),
        embeddings=embeddings,
    )
    print(f"Сохранено {len(ids)} эмбеддингов размерности {embeddings.shape[1]} в {output_path}")


if __name__ == "__main__":
    main()


