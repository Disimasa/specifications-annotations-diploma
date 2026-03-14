from __future__ import annotations

from pathlib import Path
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS_PATH = PROJECT_DIR / "data" / "ontology_grnti_embeddings.npz"


MODEL_TO_ONTOLOGY_EMB: dict[str, Path] = {
    "deepvk/USER-bge-m3": DEFAULT_EMBEDDINGS_PATH.with_name("ontology_grnti_embeddings_USER-bge-m3.npz"),
    str((PROJECT_DIR / "models" / "bi-encoder-gisnauka-trainer" / "best" / "20260314_002534").resolve()):
        DEFAULT_EMBEDDINGS_PATH.with_name("ontology_grnti_embeddings_20260314_002534.npz"),
}


def get_precomputed_embeddings_path_for_model(model_name: str) -> Optional[Path]:
    raw = (model_name or "").strip()
    if not raw:
        return None

    path = MODEL_TO_ONTOLOGY_EMB.get(raw)
    if path is not None and path.exists():
        return path

    try:
        resolved = str(Path(raw).resolve())
    except (OSError, RuntimeError, ValueError):
        resolved = ""

    if resolved:
        path = MODEL_TO_ONTOLOGY_EMB.get(resolved)
        if path is not None and path.exists():
            return path

    return None

