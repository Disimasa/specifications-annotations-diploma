from __future__ import annotations

from typing import Iterable, Sequence


def recall_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    g = set(gold)
    if not g:
        return 1.0
    pred_k = pred[:k]
    return sum(1 for p in pred_k if p in g) / float(len(g))


def precision_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    pred_k = pred[:k]
    if not pred_k:
        return 0.0
    g = set(gold)
    return sum(1 for p in pred_k if p in g) / float(len(pred_k))


def mrr_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = set(gold)
    if not g:
        return 1.0
    for i, p in enumerate(pred[:k], start=1):
        if p in g:
            return 1.0 / float(i)
    return 0.0


def ap_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = set(gold)
    if not g:
        return 1.0
    hits = 0
    s = 0.0
    for i, p in enumerate(pred[:k], start=1):
        if p in g:
            hits += 1
            s += hits / float(i)
    return s / float(len(g))


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / float(len(xs)) if xs else 0.0
