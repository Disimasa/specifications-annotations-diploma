from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from .ontology import Ontology
from .segment_filter import SegmentFilter
from .segmenter import TextSegmenter

DEFAULT_MODEL = "deepvk/USER-bge-m3"
# Модель cross-encoder для re-ranking.
# По умолчанию используем публичную DiTy/cross-encoder-russian-msmarco с HuggingFace.
DEFAULT_CROSS_ENCODER_MODEL = "DiTy/cross-encoder-russian-msmarco"


class EmbeddingAnnotator:
    def __init__(
        self,
        ontology_path: Path,
        model_name: str = DEFAULT_MODEL,
        cross_encoder_model: str | None = None,
        precomputed_embeddings_path: Path | None = None,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        # Cross-encoder для re-ranking (опционально, загружается только если используется)
        self.cross_encoder_model = cross_encoder_model
        self._cross_encoder = None

        with ontology_path.open("r", encoding="utf-8") as f:
            ontology_data = json.load(f)
        # По умолчанию используем все компетенции из онтологии
        self.ontology = Ontology(ontology_data)
        self.segmenter = TextSegmenter()
        self.segment_filter = SegmentFilter()

        # Если передан путь к заранее посчитанным эмбеддингам, пробуем их загрузить.
        # В противном случае кодируем компетенции "на лету".
        if precomputed_embeddings_path is not None and precomputed_embeddings_path.exists():
            self._competency_embeddings = self._load_precomputed_embeddings(precomputed_embeddings_path)
        else:
            self._competency_embeddings = self._encode_competencies()

    def _encode_competencies(self) -> Dict[str, torch.Tensor]:
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

    def _load_precomputed_embeddings(self, path: Path) -> Dict[str, torch.Tensor]:
        """
        Загружает заранее посчитанные эмбеддинги компетенций из .npz файла.
        Формат файла:
            ids: np.array[str] длины N
            embeddings: np.array[float32] формы (N, d)
        """
        data = np.load(path, allow_pickle=True)
        ids = data["ids"]
        embs = data["embeddings"]

        if embs.ndim != 2:
            raise ValueError(f"Ожидается 2D-массив эмбеддингов, получено {embs.shape}")

        embs_t = torch.from_numpy(embs)
        mapping: Dict[str, torch.Tensor] = {}
        for i, cid in enumerate(ids):
            cid_str = str(cid)
            mapping[cid_str] = embs_t[i]

        return mapping

    def _add_context_to_segments(self, segments: List[str], max_length: int) -> List[str]:
        """
        Добавляет контекстное окно к коротким сегментам.
        """
        segments_with_context = []

        for i, segment in enumerate(segments):
            # Если сегмент длиннее порога, оставляем как есть
            if len(segment) > max_length:
                segments_with_context.append(segment)
                continue

            # Собираем контекст: предыдущий + текущий (повторенный для взвешивания) + следующий
            context_parts = []

            # Предыдущий сегмент (если есть)
            if i > 0:
                prev_segment = segments[i - 1]
                # Ограничиваем длину предыдущего сегмента, чтобы не перегружать контекст
                if len(prev_segment) > 200:
                    prev_segment = prev_segment[:200] + "..."
                context_parts.append(prev_segment)

            # Текущий сегмент (центральный, более важный) - повторяем 2 раза для взвешивания
            context_parts.append(segment)
            context_parts.append(segment)  # Повторяем для усиления веса

            # Следующий сегмент (если есть)
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                # Ограничиваем длину следующего сегмента
                if len(next_segment) > 200:
                    next_segment = next_segment[:200] + "..."
                context_parts.append(next_segment)

            # Объединяем контекст через разделитель
            context_segment = "  ".join(context_parts)
            segments_with_context.append(context_segment)

        return segments_with_context

    def _aggregate_scores(self, scores: List[float], method: str = "sum") -> float:
        if not scores:
            return 0.0

        n = len(scores)

        if method == "max":
            return max(scores)
        if method == "mean":
            return sum(scores) / n
        if method == "median":
            sorted_scores = sorted(scores)
            if n % 2 == 0:
                return (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
            return sorted_scores[n // 2]
        if method == "weighted_mean":
            sorted_scores = sorted(scores, reverse=True)
            weights = [max(0.2, 1.0 - 0.2 * i) for i in range(len(sorted_scores))]
            weighted_sum = sum(score * weight for score, weight in zip(sorted_scores, weights))
            weight_sum = sum(weights)
            return weighted_sum / weight_sum if weight_sum > 0 else 0.0
        if method == "sum_log_count":
            return sum(scores) * float(np.log1p(n))
        if method == "mean_log_count":
            return (sum(scores) / n) * float(np.log1p(n))
        if method == "sum":
            return sum(scores)
        return sum(scores)

    def _get_cross_encoder(self) -> CrossEncoder | None:
        """Ленивая загрузка cross-encoder модели."""
        if self.cross_encoder_model is None:
            return None
        if self._cross_encoder is None:
            model_path = self.cross_encoder_model

            is_hf_model = "/" in model_path and not Path(model_path).is_absolute() and not Path(model_path).exists()

            if not is_hf_model and not Path(model_path).is_absolute():
                base_dir = Path(__file__).parent.parent
                model_path = str(base_dir / model_path)

            self._cross_encoder = CrossEncoder(model_path)
        return self._cross_encoder

    def _rerank_with_cross_encoder(self, comp, segments: List[str], candidates: List[tuple[int, float]]) -> List[tuple[int, float]]:
        cross_encoder = self._get_cross_encoder()
        if cross_encoder is None:
            return candidates

        comp_text = f"{comp.label}. {comp.description}".strip() if comp.description else comp.label

        pairs = [(comp_text, segments[idx]) for idx, _ in candidates]

        cross_scores = cross_encoder.predict(pairs)
        cross_scores_array = np.array(cross_scores)

        if self.cross_encoder_model and (
            "russian" in self.cross_encoder_model.lower()
            or "rusbeir" in self.cross_encoder_model.lower()
            or "final_model" in self.cross_encoder_model
        ):
            normalized_scores = np.clip(cross_scores_array, 0, 1)
        else:
            normalized_scores = 1 / (1 + np.exp(-cross_scores_array / 5))

        reranked = [(candidates[i][0], float(score)) for i, score in enumerate(normalized_scores)]
        reranked.sort(key=lambda p: p[1], reverse=True)
        return reranked

    def annotate(
        self,
        text: str,
        threshold: float = 0.5,
        top_k: int = 10,
        max_segment_length_for_context: int = 0,
        rerank_top_k: int = 0,
        confidence_aggregation: str = "sum",
        filter_segments: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        *,
        use_cross_encoder_doc_score: bool = False,
    ) -> List[dict]:
        # Этап: сегментация
        if progress_callback:
            progress_callback("Сегментация текста", 0.0)

        segments = self.segmenter.segment(text)

        # Фильтруем неинформативные сегменты
        if filter_segments:
            segments = self.segment_filter.filter_segments(segments)

        if not segments:
            return []

        if progress_callback:
            progress_callback("Сегментация текста", 1.0)

        # Применяем контекстное окно, если указана максимальная длина
        if max_segment_length_for_context > 0:
            if progress_callback:
                progress_callback("Добавление контекста к сегментам", 0.0)
            segments_with_context = self._add_context_to_segments(segments, max_segment_length_for_context)
            if progress_callback:
                progress_callback("Добавление контекста к сегментам", 1.0)
        else:
            segments_with_context = segments

        if progress_callback:
            progress_callback("Вычисление эмбеддингов сегментов", 0.0)

        segment_embeddings = self.model.encode(
            segments_with_context,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        if progress_callback:
            progress_callback("Вычисление эмбеддингов сегментов", 1.0)

        # Этап 1: Получаем предварительные результаты для всех компетенций по bi-encoder
        preliminary_annotations: List[dict] = []
        total_comps = len(self.ontology.competencies)
        for idx_comp, comp in enumerate(self.ontology.competencies):
            if progress_callback and total_comps:
                progress_callback("Сопоставление сегментов с компетенциями (cosine similarity)", idx_comp / total_comps)
            comp_embedding = self._competency_embeddings.get(comp.id)
            if comp_embedding is None:
                continue
            cosine_scores = util.cos_sim(segment_embeddings, comp_embedding)
            if not cosine_scores.numel():
                continue

            scores = cosine_scores.squeeze(1)
            pairs = [(idx, float(score.item())) for idx, score in enumerate(scores)]

            pairs = [(i, s) for i, s in pairs if s >= threshold]
            if not pairs:
                continue
            pairs.sort(key=lambda p: p[1], reverse=True)
            top_pairs = pairs[:top_k] if not (self.cross_encoder_model and rerank_top_k > 0) else pairs

            matches = [
                {"segment_index": idx, "score": score, "segment": segments[idx]}
                for idx, score in top_pairs
            ]

            if matches:
                scores_list = [m["score"] for m in matches]
                segment_count = len(scores_list)
                sum_score = sum(scores_list)
                mean_score = sum_score / segment_count if segment_count > 0 else 0.0
                aggregated_score = self._aggregate_scores(scores_list, confidence_aggregation)
            else:
                segment_count = 0
                sum_score = 0.0
                mean_score = 0.0
                aggregated_score = 0.0

            preliminary_annotations.append(
                {
                    "competency_id": comp.id,
                    "competency_label": comp.label,
                    "competency_full_label": comp.full_label or comp.label,
                    "doc_score": aggregated_score,
                    "max_confidence": aggregated_score,
                    "segment_count": segment_count,
                    "sum_score": sum_score,
                    "mean_score": mean_score,
                    "matches": matches,
                }
            )

        if progress_callback:
            progress_callback("Сопоставление сегментов с компетенциями (cosine similarity)", 1.0)

        preliminary_annotations.sort(key=lambda item: item.get("doc_score", 0.0), reverse=True)

        # Этап 2: Если используется re-ranking, переранжируем сегменты для top_k компетенций
        if self.cross_encoder_model and rerank_top_k > 0:
            top_competencies = preliminary_annotations[:rerank_top_k]

            final_annotations: List[dict] = []
            for ann in top_competencies:
                comp = next(c for c in self.ontology.competencies if c.id == ann["competency_id"])

                all_segments_for_comp = [(m["segment_index"], m["score"]) for m in ann["matches"]]

                reranked_pairs = self._rerank_with_cross_encoder(comp, segments, all_segments_for_comp)

                top_pairs = [(i, s) for i, s in reranked_pairs if s >= threshold]
                top_pairs = top_pairs[:top_k]

                if not top_pairs:
                    continue

                matches = [{"segment_index": idx, "score": score, "segment": segments[idx]} for idx, score in top_pairs]

                scores_list = [s for _, s in top_pairs]
                aggregated_score = self._aggregate_scores(scores_list, confidence_aggregation)

                original_bi_score = ann.get("doc_score", ann.get("max_confidence", 0.0))

                if use_cross_encoder_doc_score:
                    doc_score = aggregated_score
                    sum_score = aggregated_score
                    mean_score = aggregated_score
                else:
                    doc_score = original_bi_score
                    sum_score = ann.get("sum_score", original_bi_score)
                    mean_score = ann.get("mean_score", original_bi_score)

                final_annotations.append(
                    {
                        "competency_id": comp.id,
                        "competency_label": comp.label,
                        "competency_full_label": comp.full_label or comp.label,
                        "doc_score": doc_score,
                        "max_confidence": doc_score,
                        "segment_count": ann.get("segment_count", 0),
                        "sum_score": sum_score,
                        "mean_score": mean_score,
                        "matches": matches,
                    }
                )

            final_annotations.extend(preliminary_annotations[rerank_top_k:])
            final_annotations.sort(key=lambda item: item.get("doc_score", 0.0), reverse=True)
            return final_annotations

        return preliminary_annotations


def annotate_document(
    text_path: Path,
    ontology_path: Path,
    threshold: float = 0.5,
    top_k: int = 10,
    max_segment_length_for_context: int = 0,
    rerank_top_k: int = 0,
    cross_encoder_model: str | None = None,
    confidence_aggregation: str = "sum",
    filter_segments: bool = True,
    precomputed_embeddings_path: Path | None = None,
    use_cross_encoder_doc_score: bool = False,
) -> List[dict]:
    if rerank_top_k > 0 and cross_encoder_model is None:
        cross_encoder_model = DEFAULT_CROSS_ENCODER_MODEL

    if rerank_top_k == 0:
        cross_encoder_model = None

    annotator = EmbeddingAnnotator(
        ontology_path=ontology_path,
        cross_encoder_model=cross_encoder_model,
        precomputed_embeddings_path=precomputed_embeddings_path,
    )
    text = text_path.read_text(encoding="utf-8")
    return annotator.annotate(
        text,
        threshold=threshold,
        top_k=top_k,
        max_segment_length_for_context=max_segment_length_for_context,
        rerank_top_k=rerank_top_k,
        confidence_aggregation=confidence_aggregation,
        filter_segments=filter_segments,
        use_cross_encoder_doc_score=use_cross_encoder_doc_score,
    )

