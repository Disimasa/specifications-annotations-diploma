"""
Иерархический batch sampler для GRNTI (leaf/parent/grand) + curriculum + safe-hard + баланс по grand/leaf.

Совместим с SentenceTransformerTrainer: наследует DefaultBatchSampler, поддерживает set_epoch.

Пошаговое описание этапов: README_hierarchical_grnti_batch_sampler.md (рядом с этим файлом).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import BatchSampler

from sentence_transformers import SentenceTransformer
from sentence_transformers.sampler import DefaultBatchSampler

try:
    from datasets import Dataset
except ImportError:
    Dataset = None  # type: ignore


def _parse_triplet(s: str) -> Tuple[float, float, float]:
    if s.strip().lower() in {"random", "rand", "none"}:
        # Спец-режим: не гоним батч к целевым долям, выбираем по базовым весам/фильтрам.
        return float("nan"), float("nan"), float("nan")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Ожидается три числа через запятую (far,mid,hard), получено: {s!r}")
    a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
    tot = a + b + c
    if tot <= 0:
        raise ValueError(f"Сумма долей curriculum должна быть > 0: {s!r}")
    return a / tot, b / tot, c / tot


def relation_type(
    leaf_i: str,
    grand_i: str,
    parent_i: str,
    leaf_j: str,
    grand_j: str,
    parent_j: str,
) -> str:
    """Тип отношения positive_j как in-batch negative для anchor i."""
    if leaf_i == leaf_j:
        return "dup"
    if grand_i != grand_j:
        return "far"
    if parent_i != parent_j:
        return "mid"
    return "hard"


@dataclass
class _RowMeta:
    doc_id: str
    leaf: str
    parent: str
    grand: str
    gold_leaves: frozenset[str]
    text1: str
    text2: str


class HierarchicalGrntiBatchSampler(DefaultBatchSampler):
    """
    Формирует батчи под MultipleNegatives / Cached* losses:
    - не более одного сэмпла на doc_id;
    - не более одного одинакового leaf (positive) в батче;
    - multi-label: positive другого сэмпла не входит в gold_set документа anchor;
    - доли directed пар (far/mid/hard) к концу батча стремятся к curriculum для текущей эпохи;
    - safe-hard: для пар типа hard — проверка guide (relative margin), иначе кандидат отбрасывается;
    - выбор кандидатов взвешен: редкие leaf и недопредставленные grand.
    """

    def __init__(
        self,
        dataset: "Dataset",
        batch_size: int,
        drop_last: bool,
        valid_label_columns: Optional[List[str]] = None,
        generator: Optional[torch.Generator] = None,
        seed: int = 0,
        *,
        guide_model: Optional[SentenceTransformer] = None,
        curriculum_epoch1: Tuple[float, float, float] = (0.8, 0.2, 0.0),
        curriculum_epoch2: Tuple[float, float, float] = (0.6, 0.3, 0.1),
        curriculum_epoch3plus: Tuple[float, float, float] = (0.45, 0.35, 0.2),
        relative_margin: float = 0.05,
        leaf_balance_power: float = 0.5,
        grand_balance_weight: float = 1.0,
        max_scored_candidates: int = 256,
        fallback_relaxed: bool = True,
        enable_diagnostics: bool = True,
    ) -> None:
        # Не удалять метаданные GRNTI даже если valid_label_columns пересекается с именами колонок.
        _grnti_meta_cols = {"doc_id", "leaf", "parent", "grand", "doc_gold_leaves"}
        if label_columns := set(dataset.column_names) & set(valid_label_columns or []):
            label_columns -= _grnti_meta_cols
            if label_columns:
                dataset = dataset.remove_columns(list(label_columns))
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )
        self.dataset = dataset
        self.guide_model = guide_model
        self.curriculum_epoch1 = curriculum_epoch1
        self.curriculum_epoch2 = curriculum_epoch2
        self.curriculum_epoch3plus = curriculum_epoch3plus
        self.relative_margin = float(relative_margin)
        self.leaf_balance_power = float(leaf_balance_power)
        self.grand_balance_weight = float(grand_balance_weight)
        self.max_scored_candidates = int(max_scored_candidates)
        self.fallback_relaxed = bool(fallback_relaxed)
        self.enable_diagnostics = bool(enable_diagnostics)

        required = {"text1", "text2", "doc_id", "leaf", "parent", "grand", "doc_gold_leaves"}
        missing = required - set(self.dataset.column_names)
        if missing:
            raise ValueError(f"Датасету не хватает колонок для hierarchical sampler: {sorted(missing)}")

        self._metas: List[_RowMeta] = []
        leaf_cnt: Dict[str, int] = {}
        grand_cnt: Dict[str, int] = {}
        for i in range(len(self.dataset)):
            row = self.dataset[i]
            doc_id = str(row["doc_id"])
            leaf = str(row["leaf"])
            parent = str(row["parent"])
            grand = str(row["grand"])
            raw_gold = str(row["doc_gold_leaves"] or "")
            gold_set = frozenset(c.strip() for c in raw_gold.split(";") if c.strip())
            self._metas.append(
                _RowMeta(
                    doc_id=doc_id,
                    leaf=leaf,
                    parent=parent,
                    grand=grand,
                    gold_leaves=gold_set,
                    text1=str(row["text1"]),
                    text2=str(row["text2"]),
                )
            )
            leaf_cnt[leaf] = leaf_cnt.get(leaf, 0) + 1
            grand_cnt[grand] = grand_cnt.get(grand, 0) + 1

        self._leaf_cnt = leaf_cnt
        self._grand_cnt = grand_cnt
        max_leaf = max(leaf_cnt.values()) if leaf_cnt else 1
        max_grand = max(grand_cnt.values()) if grand_cnt else 1
        self._base_w: List[float] = []
        for m in self._metas:
            lw = (max_leaf / max(1, leaf_cnt.get(m.leaf, 1))) ** self.leaf_balance_power
            gw = (max_grand / max(1, grand_cnt.get(m.grand, 1))) ** self.grand_balance_weight
            self._base_w.append(float(lw * gw))

        self._emb_cache_seg: Dict[int, torch.Tensor] = {}
        self._emb_cache_pos: Dict[int, torch.Tensor] = {}

        self.diagnostics_last_epoch: Dict[str, float] = {}
        self._diag_edges_far = 0
        self._diag_edges_mid = 0
        self._diag_edges_hard = 0
        self._diag_reject_multilabel = 0
        self._diag_reject_dup_leaf = 0
        self._diag_reject_dup_doc = 0
        self._diag_reject_guide = 0
        self._diag_batches = 0
        self._diag_fallbacks = 0
        self._diag_unsafe_hard_relaxed = 0

    def _training_epoch_1based(self) -> int:
        # HF Trainer: set_epoch(0) на первой эпохе, затем инкремент
        return int(self.epoch) + 1

    def _curriculum_targets(self) -> Tuple[float, float, float]:
        e = self._training_epoch_1based()
        if e <= 1:
            return self.curriculum_epoch1
        if e == 2:
            return self.curriculum_epoch2
        return self.curriculum_epoch3plus

    def _encode_cache(self, idx: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx not in self._emb_cache_seg:
            m = self._metas[idx]
            if self.guide_model is None:
                raise RuntimeError("guide_model required for safe-hard check")
            with torch.inference_mode():
                seg_e = self.guide_model.encode(
                    [m.text1],
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    batch_size=1,
                    show_progress_bar=False,
                )
                pos_e = self.guide_model.encode(
                    [m.text2],
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    batch_size=1,
                    show_progress_bar=False,
                )
            self._emb_cache_seg[idx] = seg_e.detach().to("cpu")
            self._emb_cache_pos[idx] = pos_e.detach().to("cpu")
        return self._emb_cache_seg[idx].to(device), self._emb_cache_pos[idx].to(device)

    def _safe_hard_ok(self, batch_idx: List[int], j: int, device: torch.device) -> bool:
        if self.guide_model is None:
            return True
        self.guide_model.eval()
        mj = self._metas[j]
        ej_seg, ej_pos = self._encode_cache(j, device)
        for i in batch_idx:
            mi = self._metas[i]
            rij = relation_type(mi.leaf, mi.grand, mi.parent, mj.leaf, mj.grand, mj.parent)
            rji = relation_type(mj.leaf, mj.grand, mj.parent, mi.leaf, mi.grand, mi.parent)
            if rij == "hard":
                ei_seg, ei_pos = self._encode_cache(i, device)
                s_neg = torch.nn.functional.cosine_similarity(ei_seg, ej_pos).item()
                s_pos = torch.nn.functional.cosine_similarity(ei_seg, ei_pos).item()
                if s_neg >= s_pos * (1.0 - self.relative_margin):
                    self._diag_reject_guide += 1
                    return False
            if rji == "hard":
                ej_seg2, ej_pos2 = ej_seg, ej_pos
                ei_seg2, ei_pos2 = self._encode_cache(i, device)
                s_neg = torch.nn.functional.cosine_similarity(ej_seg2, ei_pos2).item()
                s_pos = torch.nn.functional.cosine_similarity(ej_seg2, ej_pos2).item()
                if s_neg >= s_pos * (1.0 - self.relative_margin):
                    self._diag_reject_guide += 1
                    return False
        return True

    def _multilabel_ok(self, batch_idx: List[int], j: int) -> bool:
        mj = self._metas[j]
        for i in batch_idx:
            mi = self._metas[i]
            if mj.leaf in mi.gold_leaves or mi.leaf in mj.gold_leaves:
                self._diag_reject_multilabel += 1
                return False
        return True

    def _basic_ok(self, batch_idx: List[int], j: int) -> bool:
        mj = self._metas[j]
        docs = {self._metas[i].doc_id for i in batch_idx}
        leaves = {self._metas[i].leaf for i in batch_idx}
        if mj.doc_id in docs:
            self._diag_reject_dup_doc += 1
            return False
        if mj.leaf in leaves:
            self._diag_reject_dup_leaf += 1
            return False
        return True

    def _edge_delta(self, batch_idx: List[int], j: int) -> Tuple[int, int, int]:
        """Добавляем directed edges (i->j) и (j->i) для i в batch."""
        df = dm = dh = 0
        mj = self._metas[j]
        for i in batch_idx:
            mi = self._metas[i]
            for a, b in (
                (mi, mj),
                (mj, mi),
            ):
                t = relation_type(a.leaf, a.grand, a.parent, b.leaf, b.grand, b.parent)
                if t == "dup":
                    continue
                if t == "far":
                    df += 1
                elif t == "mid":
                    dm += 1
                else:
                    dh += 1
        return df, dm, dh

    def _score_candidate(
        self,
        batch_idx: List[int],
        j: int,
        target: Tuple[float, float, float],
        cf: int,
        cm: int,
        ch: int,
    ) -> float:
        if any(math.isnan(x) for x in target):
            return 0.0
        df, dm, dh = self._edge_delta(batch_idx, j)
        nf, nm, nh = cf + df, cm + dm, ch + dh
        tot = nf + nm + nh
        if tot == 0:
            return float("inf")
        err = (
            abs(nf / tot - target[0])
            + abs(nm / tot - target[1])
            + abs(nh / tot - target[2])
        )
        return -err

    def _reset_diag(self) -> None:
        self._diag_edges_far = 0
        self._diag_edges_mid = 0
        self._diag_edges_hard = 0
        self._diag_reject_multilabel = 0
        self._diag_reject_dup_leaf = 0
        self._diag_reject_dup_doc = 0
        self._diag_reject_guide = 0
        self._diag_batches = 0
        self._diag_fallbacks = 0
        self._diag_unsafe_hard_relaxed = 0

    def _finalize_batch_diag(self, batch_idx: List[int]) -> None:
        cf = cm = ch = 0
        for ii in range(len(batch_idx)):
            for jj in range(len(batch_idx)):
                if ii == jj:
                    continue
                i = batch_idx[ii]
                j = batch_idx[jj]
                mi = self._metas[i]
                mj = self._metas[j]
                t = relation_type(mi.leaf, mi.grand, mi.parent, mj.leaf, mj.grand, mj.parent)
                if t == "far":
                    cf += 1
                elif t == "mid":
                    cm += 1
                elif t == "hard":
                    ch += 1
        self._diag_edges_far += cf
        self._diag_edges_mid += cm
        self._diag_edges_hard += ch
        self._diag_batches += 1

    def __iter__(self) -> Iterator[List[int]]:
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        self._reset_diag()
        device = self.guide_model.device if self.guide_model is not None else torch.device("cpu")

        n = len(self.dataset)
        perm = torch.randperm(n, generator=self.generator).tolist()
        remaining = dict.fromkeys(perm)
        target = self._curriculum_targets()

        while remaining:
            batch: List[int] = []
            cf = cm = ch = 0
            while len(batch) < self.batch_size:
                pool = [idx for idx in remaining if idx not in batch]
                if not pool:
                    break

                w = torch.tensor([self._base_w[i] for i in pool], dtype=torch.double)
                w = w / w.sum().clamp_min(1e-12)
                cand_count = min(len(pool), self.max_scored_candidates)
                pick = torch.multinomial(w, cand_count, replacement=False, generator=self.generator).tolist()
                candidates = [pool[k] for k in pick]

                best_j: Optional[int] = None
                best_score = float("-inf")
                for j in candidates:
                    if not self._basic_ok(batch, j):
                        continue
                    if not self._multilabel_ok(batch, j):
                        continue
                    if not self._safe_hard_ok(batch, j, device):
                        continue
                    sc = self._score_candidate(batch, j, target, cf, cm, ch)
                    if sc > best_score:
                        best_score = sc
                        best_j = j

                if best_j is None and self.fallback_relaxed:
                    order = torch.randperm(len(pool), generator=self.generator).tolist()
                    for k in order:
                        j = pool[k]
                        if not self._basic_ok(batch, j):
                            continue
                        if not self._multilabel_ok(batch, j):
                            continue
                        if not self._safe_hard_ok(batch, j, device):
                            continue
                        best_j = j
                        self._diag_fallbacks += 1
                        break

                if best_j is None and self.guide_model is not None:
                    order = torch.randperm(len(pool), generator=self.generator).tolist()
                    for k in order:
                        j = pool[k]
                        if not self._basic_ok(batch, j):
                            continue
                        if not self._multilabel_ok(batch, j):
                            continue
                        best_j = j
                        self._diag_unsafe_hard_relaxed += 1
                        break

                if best_j is None:
                    break

                df, dm, dh = self._edge_delta(batch, best_j)
                cf += df
                cm += dm
                ch += dh
                batch.append(best_j)

            if len(batch) == self.batch_size:
                self._finalize_batch_diag(batch)
                yield batch
            elif batch and not self.drop_last:
                self._finalize_batch_diag(batch)
                yield batch

            for idx in batch:
                if idx in remaining:
                    del remaining[idx]

        tot_e = self._diag_edges_far + self._diag_edges_mid + self._diag_edges_hard
        self.diagnostics_last_epoch = {
            "batches": float(self._diag_batches),
            "edges_far": float(self._diag_edges_far),
            "edges_mid": float(self._diag_edges_mid),
            "edges_hard": float(self._diag_edges_hard),
            "edge_far_frac": float(self._diag_edges_far / tot_e) if tot_e else 0.0,
            "edge_mid_frac": float(self._diag_edges_mid / tot_e) if tot_e else 0.0,
            "edge_hard_frac": float(self._diag_edges_hard / tot_e) if tot_e else 0.0,
            "reject_multilabel": float(self._diag_reject_multilabel),
            "reject_dup_leaf": float(self._diag_reject_dup_leaf),
            "reject_dup_doc": float(self._diag_reject_dup_doc),
            "reject_guide": float(self._diag_reject_guide),
            "fallbacks": float(self._diag_fallbacks),
            "unsafe_hard_relaxed": float(self._diag_unsafe_hard_relaxed),
            "curriculum_epoch": float(self._training_epoch_1based()),
        }
        if self.enable_diagnostics and tot_e > 0:
            print(
                f"[hierarchical_sampler] epoch={self._training_epoch_1based()} "
                f"target_far,mid,hard={target[0]:.2f},{target[1]:.2f},{target[2]:.2f} | "
                f"edges far/mid/hard={self.diagnostics_last_epoch['edge_far_frac']:.2f}/"
                f"{self.diagnostics_last_epoch['edge_mid_frac']:.2f}/"
                f"{self.diagnostics_last_epoch['edge_hard_frac']:.2f} | "
                f"rej ml/dupL/dupD/guide={self._diag_reject_multilabel}/"
                f"{self._diag_reject_dup_leaf}/{self._diag_reject_dup_doc}/{self._diag_reject_guide} | "
                f"fallbacks={self._diag_fallbacks} unsafe_hard_relaxed={self._diag_unsafe_hard_relaxed}"
            )

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class HierarchicalGrntiBatchSamplerFactory:
    """
    Callable для SentenceTransformerTrainingArguments.batch_sampler.

    Должен быть классом на уровне модуля (не вложенная функция): иначе на Windows при
    multiprocessing / pickle падает «Can't pickle local object ... factory».
    """

    def __init__(
        self,
        *,
        guide_model: Optional[SentenceTransformer],
        curriculum_epoch1: str,
        curriculum_epoch2: str,
        curriculum_epoch3plus: str,
        relative_margin: float,
        leaf_balance_power: float,
        grand_balance_weight: float,
        max_scored_candidates: int,
        enable_diagnostics: bool,
        fallback_relaxed: bool = True,
    ) -> None:
        self.guide_model = guide_model
        self.curriculum_epoch1 = _parse_triplet(curriculum_epoch1)
        self.curriculum_epoch2 = _parse_triplet(curriculum_epoch2)
        self.curriculum_epoch3plus = _parse_triplet(curriculum_epoch3plus)
        self.relative_margin = float(relative_margin)
        self.leaf_balance_power = float(leaf_balance_power)
        self.grand_balance_weight = float(grand_balance_weight)
        self.max_scored_candidates = int(max_scored_candidates)
        self.enable_diagnostics = bool(enable_diagnostics)
        self.fallback_relaxed = bool(fallback_relaxed)

    def __call__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: Optional[List[str]] = None,
        generator: Optional[torch.Generator] = None,
        seed: int = 0,
    ) -> BatchSampler:
        # Trainer передаёт collator.valid_label_columns — не удаляем метаданные GRNTI.
        _ = valid_label_columns
        return HierarchicalGrntiBatchSampler(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=[],
            generator=generator,
            seed=seed,
            guide_model=self.guide_model,
            curriculum_epoch1=self.curriculum_epoch1,
            curriculum_epoch2=self.curriculum_epoch2,
            curriculum_epoch3plus=self.curriculum_epoch3plus,
            relative_margin=self.relative_margin,
            leaf_balance_power=self.leaf_balance_power,
            grand_balance_weight=self.grand_balance_weight,
            max_scored_candidates=self.max_scored_candidates,
            enable_diagnostics=self.enable_diagnostics,
            fallback_relaxed=self.fallback_relaxed,
        )


def create_hierarchical_batch_sampler_factory(
    *,
    guide_model: Optional[SentenceTransformer],
    curriculum_epoch1: str,
    curriculum_epoch2: str,
    curriculum_epoch3plus: str,
    relative_margin: float,
    leaf_balance_power: float,
    grand_balance_weight: float,
    max_scored_candidates: int,
    enable_diagnostics: bool,
    fallback_relaxed: bool = True,
) -> HierarchicalGrntiBatchSamplerFactory:
    """Обёртка для совместимости; возвращает picklable-экземпляр фабрики."""
    return HierarchicalGrntiBatchSamplerFactory(
        guide_model=guide_model,
        curriculum_epoch1=curriculum_epoch1,
        curriculum_epoch2=curriculum_epoch2,
        curriculum_epoch3plus=curriculum_epoch3plus,
        relative_margin=relative_margin,
        leaf_balance_power=leaf_balance_power,
        grand_balance_weight=grand_balance_weight,
        max_scored_candidates=max_scored_candidates,
        enable_diagnostics=enable_diagnostics,
        fallback_relaxed=fallback_relaxed,
    )
