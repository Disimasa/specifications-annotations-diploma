from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer, util

from .ontology import Ontology
from .segmenter import TextSegmenter

DEFAULT_MODEL = "ai-forever/sbert_large_nlu_ru"


class EmbeddingAnnotator:
    def __init__(self, ontology_path: Path, model_name: str = DEFAULT_MODEL) -> None:
        self.model = SentenceTransformer(model_name)
        with ontology_path.open("r", encoding="utf-8") as f:
            ontology_data = json.load(f)
        # По умолчанию используем все компетенции из онтологии
        self.ontology = Ontology(ontology_data)
        self.segmenter = TextSegmenter()
        self._competency_embeddings = self._encode_competencies()

    def _encode_competencies(self) -> Dict[str, List[float]]:
        texts = [
            f"{comp.label}. {comp.description}".strip()
            if comp.description
            else comp.label
            for comp in self.ontology.competencies
        ]
        if not texts:
            return {}
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return {comp.id: embeddings[i] for i, comp in enumerate(self.ontology.competencies)}

    def annotate(self, text: str, threshold: float = 0.5, top_k: int = 10) -> List[dict]:
        segments = self.segmenter.segment(text)
        # for seg in segments:
        #     print(seg)
        if not segments:
            return []

        segment_embeddings = self.model.encode(
            segments,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        annotations: List[dict] = []
        for comp in self.ontology.competencies:
            comp_embedding = self._competency_embeddings.get(comp.id)
            if comp_embedding is None:
                continue
            cosine_scores = util.cos_sim(segment_embeddings, comp_embedding)
            if not cosine_scores.numel():
                continue

            # Преобразуем в одномерный список (по всем сегментам)
            scores = cosine_scores.squeeze(1)
            pairs = [
                (idx, float(score.item()))
                for idx, score in enumerate(scores)
            ]
            # Отфильтровать по порогу и взять top_k
            pairs = [(i, s) for i, s in pairs if s >= threshold]
            if not pairs:
                continue
            pairs.sort(key=lambda p: p[1], reverse=True)
            top_pairs = pairs[:top_k]

            matches = [
                {
                    "segment_index": idx,
                    "score": score,
                    "segment": segments[idx],
                }
                for idx, score in top_pairs
            ]

            annotations.append(
                {
                    "competency_id": comp.id,
                    "competency_label": comp.label,
                    "max_confidence": matches[0]["score"],
                    "matches": matches,
                }
            )

        annotations.sort(key=lambda item: item["max_confidence"], reverse=True)
        return annotations


def annotate_document(text_path: Path, ontology_path: Path, threshold: float = 0.5, top_k: int = 10) -> List[dict]:
    annotator = EmbeddingAnnotator(ontology_path=ontology_path)
    text = text_path.read_text(encoding="utf-8")
    return annotator.annotate(text, threshold=threshold, top_k=top_k)

