from __future__ import annotations

"""
Сбор "silver" тестовой выборки (авторазметка) на основе уже имеющихся ТЗ.

Важно: это НЕ gold-эталон. Метки генерируются текущей системой аннотации и
предназначены для быстрого регрессионного прогона и как черновик для ручной проверки.

Выход: data/gold/test_set_silver.jsonl
Одна строка = один документ:
{
  "doc_id": "...",
  "text_path": "...",
  "gold_competency_ids": [...],   # здесь "gold" в смысле целевых меток (silver)
  "meta": { ... параметры ... }
}
"""

import json
from pathlib import Path
from typing import Dict, List

from annotation.annotator import annotate_document


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEXTS_DIR = PROJECT_DIR / "data" / "specifications" / "texts"
ONTOLOGY_PATH = PROJECT_DIR / "data" / "ontology_grnti_with_llm.json"
EMB_PATH = PROJECT_DIR / "data" / "ontology_grnti_embeddings.npz"
OUT_PATH = PROJECT_DIR / "data" / "gold" / "test_set_silver.jsonl"


def load_existing_index(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    index: Dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        doc_id = obj.get("doc_id")
        if isinstance(doc_id, str):
            index[doc_id] = obj
    return index


def main() -> None:
    if not TEXTS_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка с текстами: {TEXTS_DIR}")
    if not ONTOLOGY_PATH.exists():
        raise FileNotFoundError(f"Не найдена онтология: {ONTOLOGY_PATH}")
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Не найдены эмбеддинги онтологии: {EMB_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_index(OUT_PATH)

    # Параметры авторазметки (можно менять под задачу)
    threshold = 0.5
    top_k = 10
    top_n = 20
    rerank_top_k = 20
    confidence_aggregation = "sum"

    all_txts = sorted(TEXTS_DIR.glob("*.txt"))
    if not all_txts:
        raise RuntimeError(f"В папке нет .txt файлов: {TEXTS_DIR}")

    rows: List[dict] = []
    for path in all_txts:
        doc_id = path.stem
        if doc_id in existing:
            rows.append(existing[doc_id])
            continue

        annotations = annotate_document(
            text_path=path,
            ontology_path=ONTOLOGY_PATH,
            threshold=threshold,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            confidence_aggregation=confidence_aggregation,
            precomputed_embeddings_path=EMB_PATH,
        )

        predicted_ids = [a["competency_id"] for a in annotations[:top_n] if "competency_id" in a]

        row = {
            "doc_id": doc_id,
            "text_path": str(path),
            "gold_competency_ids": predicted_ids,
            "meta": {
                "kind": "silver_autolabel",
                "threshold": threshold,
                "top_k": top_k,
                "top_n": top_n,
                "rerank_top_k": rerank_top_k,
                "confidence_aggregation": confidence_aggregation,
                "ontology_path": str(ONTOLOGY_PATH),
                "precomputed_embeddings_path": str(EMB_PATH),
            },
        }
        rows.append(row)

        # сохраняем прогресс инкрементально
        OUT_PATH.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    print(f"Готово. Документов в test_set_silver: {len(rows)}")
    print(f"Файл: {OUT_PATH}")


if __name__ == "__main__":
    main()

