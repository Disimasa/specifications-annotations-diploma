from __future__ import annotations

"""
Предварительный расчёт эмбеддингов для онтологии ГРНТИ.

- Читает ontology_grnti_with_llm.json (nodes + links, с полями full_label и llm_description)
- Для всех компетенций (id начинается с http://example.org/competencies#) готовит текст:
    text_for_embedding = "{label}. {llm_description}. {full_label}"
  (части, которых нет, пропускаются)
- Считает эмбеддинги с помощью SentenceTransformer
- Сохраняет результат в data/ontology_grnti_embeddings.npz
  (ids: список id, embeddings: массив float32 [N, D])
"""

import json
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
OUTPUT_PATH = PROJECT_DIR / "data" / "ontology_grnti_embeddings.npz"

COMP_PREFIX = "http://example.org/competencies#"

# Модель по умолчанию — та же, что и в пайплайне (можно поменять при необходимости)
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


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Не найден входной файл {INPUT_PATH}")

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])

    # Отбираем все компетенции (включая не-листовые; фильтрация листов при необходимости — на этапе использования)
    comp_nodes = [n for n in nodes if str(n.get("id", "")).startswith(COMP_PREFIX)]

    ids: List[str] = []
    texts: List[str] = []

    for node in comp_nodes:
        node_id = node["id"]
        text = build_text_for_embedding(node)
        if not text:
            # Если по какой-то причине текст пустой, пропускаем — эмбеддинг для пустоты не нужен
            continue
        ids.append(node_id)
        texts.append(text)

    if not ids:
        raise RuntimeError("Не найдено ни одной компетенции с непустым текстом для эмбеддинга.")

    print(f"Всего компетенций для эмбеддингов: {len(ids)}")

    model = SentenceTransformer(DEFAULT_MODEL_NAME)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype(np.float32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        ids=np.array(ids, dtype=object),
        embeddings=embeddings,
    )
    print(f"Сохранено {len(ids)} эмбеддингов размерности {embeddings.shape[1]} в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

