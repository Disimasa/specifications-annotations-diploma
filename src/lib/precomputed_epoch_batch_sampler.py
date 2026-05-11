"""
Предгенерированные батчи по эпохам (см. scripts/train/generate_hierarchical_batches.py).

Совместим с SentenceTransformerTrainer: DefaultBatchSampler + set_epoch.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from sentence_transformers.sampler import DefaultBatchSampler

try:
    from datasets import Dataset
except ImportError:
    Dataset = None  # type: ignore


def load_precomputed_batches_payload(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Файл предбатчей не найден: {p}")
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def _validate_indices(dataset_len: int, batches_by_epoch: List[List[List[int]]]) -> None:
    for ei, epoch_batches in enumerate(batches_by_epoch):
        for bi, batch in enumerate(epoch_batches):
            for idx in batch:
                if idx < 0 or idx >= dataset_len:
                    raise ValueError(
                        f"Некорректный индекс в предбатчах: epoch={ei} batch={bi} "
                        f"index={idx}, len(dataset)={dataset_len}"
                    )


class PrecomputedEpochBatchSampler(DefaultBatchSampler):
    """
    На каждой эпохе (set_epoch) отдаёт заранее сохранённый список батчей (индексы в train Dataset).

    Требование: число батчей одинаково для каждой записанной эпохи (иначе len(DataLoader) нестабилен).
    Если num_train_epochs больше, чем len(batches_by_epoch), последний список батчей повторяется.
    """

    def __init__(
        self,
        dataset: "Dataset",
        batch_size: int,
        drop_last: bool,
        batches_by_epoch: List[List[List[int]]],
        valid_label_columns: Optional[List[str]] = None,
        generator: Optional[torch.Generator] = None,
        seed: int = 0,
    ) -> None:
        if not batches_by_epoch:
            raise ValueError("batches_by_epoch пуст")
        lens = [len(eb) for eb in batches_by_epoch]
        if len(set(lens)) > 1:
            raise ValueError(
                "В файле предбатчей разное число батчей по эпохам "
                f"{lens}. Перегенерируйте файл — иначе len(dataloader) не согласуется с Trainer."
            )
        if label_columns := set(dataset.column_names) & set(valid_label_columns or []):
            dataset = dataset.remove_columns(list(label_columns))
        self.dataset = dataset
        self.batches_by_epoch = batches_by_epoch
        self._warned_epoch_repeat = False
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )

    def __iter__(self):
        if self.generator is not None and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        n_ep = len(self.batches_by_epoch)
        idx = int(self.epoch)
        if idx >= n_ep:
            idx = n_ep - 1
            if n_ep > 0 and not self._warned_epoch_repeat:
                warnings.warn(
                    f"Эпох обучения больше, чем в файле предбатчей ({n_ep}); "
                    f"начиная с epoch>={n_ep} повторяется последний список батчей.",
                    stacklevel=2,
                )
                self._warned_epoch_repeat = True
        elif idx < 0:
            idx = 0

        for batch in self.batches_by_epoch[idx]:
            yield list(batch)

    def __len__(self) -> int:
        return len(self.batches_by_epoch[0])


class PrecomputedEpochBatchSamplerFactory:
    """
    Callable для SentenceTransformerTrainingArguments.batch_sampler (picklable, не вложенная функция).
    """

    def __init__(
        self,
        precomputed_path: Path | str,
        fn_pair_frac_max: float | None = None,
    ) -> None:
        path = Path(precomputed_path)
        payload = load_precomputed_batches_payload(path)
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError(f"Неизвестный format_version в {path}: {payload.get('format_version')}")

        self.batches_by_epoch: List[List[List[int]]] = payload["batches_by_epoch"]
        self.stored_size = payload.get("dataset_size")
        file_args = payload.get("args") or {}
        self.file_bs = int(file_args.get("batch_size", 0))
        self.file_drop_last = bool(file_args.get("drop_last", False))
        self.fn_pair_frac_max = float(fn_pair_frac_max) if fn_pair_frac_max is not None else None

    def __call__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: Optional[List[str]] = None,
        generator: Optional[torch.Generator] = None,
        seed: int = 0,
    ):
        if self.stored_size is not None and len(dataset) != int(self.stored_size):
            raise ValueError(
                f"Размер train_dataset ({len(dataset)}) не совпадает с записанным в файле "
                f"({self.stored_size}). Соберите датасет так же, как при generate_hierarchical_batches "
                "(те же CSV, ontology, seed shuffle, --max-train-samples)."
            )
        if self.file_bs and int(batch_size) != self.file_bs:
            raise ValueError(
                f"--batch-size ({batch_size}) должен совпадать с batch_size в файле предбатчей ({self.file_bs})."
            )
        if bool(drop_last) != self.file_drop_last:
            raise ValueError(
                f"dataloader_drop_last в тренере ({drop_last}) не совпадает с drop_last при генерации "
                f"({self.file_drop_last}). Задайте в SentenceTransformerTrainingArguments(dataloader_drop_last=...) "
                "или перегенерируйте батчи."
            )
        _validate_indices(len(dataset), self.batches_by_epoch)
        if self.fn_pair_frac_max is not None:
            # Фильтруем батчи по FN-парам (anchor->candidate), где leaf(candidate) содержится в doc_gold_leaves(anchor).
            required_cols = {"leaf", "doc_gold_leaves"}
            cols = set(getattr(dataset, "column_names", []) or [])
            missing = required_cols - cols
            if missing:
                raise ValueError(
                    f"Для FN-фильтрации нужен датасет с колонками {sorted(required_cols)}, "
                    f"но в train_dataset колонки: {sorted(cols)}. Missing={sorted(missing)}"
                )

            leaf_cache: dict[int, str] = {}
            gold_cache: dict[int, frozenset[str]] = {}

            def get_leaf(idx: int) -> str:
                if idx not in leaf_cache:
                    leaf_cache[idx] = str(dataset[idx]["leaf"])
                return leaf_cache[idx]

            def get_gold(idx: int) -> frozenset[str]:
                if idx not in gold_cache:
                    raw = str(dataset[idx].get("doc_gold_leaves") or "")
                    gold_cache[idx] = frozenset(c.strip() for c in raw.split(";") if c.strip())
                return gold_cache[idx]

            filtered_by_epoch: List[List[List[int]]] = []
            for epoch_batches in self.batches_by_epoch:
                new_epoch_batches: List[List[int]] = []
                for batch in epoch_batches:
                    bsz = len(batch)
                    if bsz <= 1:
                        new_epoch_batches.append(batch)
                        continue
                    total_pairs = bsz * (bsz - 1)
                    fn_pairs = 0
                    # FN-проверка: (a -> c), a!=c, leaf_c in gold_sets[a]
                    for a in range(bsz):
                        gold_a = get_gold(batch[a])
                        for c in range(bsz):
                            if a == c:
                                continue
                            if get_leaf(batch[c]) in gold_a:
                                fn_pairs += 1
                        # quick break if already above threshold (optional)
                        if float(fn_pairs) / float(total_pairs) > self.fn_pair_frac_max:
                            break
                    fn_pair_frac = float(fn_pairs) / float(total_pairs) if total_pairs else 0.0
                    if fn_pair_frac <= self.fn_pair_frac_max:
                        new_epoch_batches.append(batch)

                filtered_by_epoch.append(new_epoch_batches)

            # Требование PrecomputedEpochBatchSampler: одинаковая длина batches_by_epoch по эпохам.
            min_len = min(len(eb) for eb in filtered_by_epoch) if filtered_by_epoch else 0
            if min_len <= 0:
                raise RuntimeError(
                    f"FN-фильтрация {self.fn_pair_frac_max} оставила 0 батчей (или все эпохи полностью вырезаны)."
                )
            filtered_by_epoch = [eb[:min_len] for eb in filtered_by_epoch]

            return PrecomputedEpochBatchSampler(
                dataset,
                batch_size=int(batch_size),
                drop_last=bool(drop_last),
                batches_by_epoch=filtered_by_epoch,
                valid_label_columns=valid_label_columns,
                generator=generator,
                seed=seed,
            )

        return PrecomputedEpochBatchSampler(
            dataset,
            batch_size=int(batch_size),
            drop_last=bool(drop_last),
            batches_by_epoch=self.batches_by_epoch,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )


def create_precomputed_batch_sampler_factory(
    precomputed_path: Path | str,
    fn_pair_frac_max: float | None = None,
) -> PrecomputedEpochBatchSamplerFactory:
    """Обёртка для совместимости."""
    return PrecomputedEpochBatchSamplerFactory(precomputed_path, fn_pair_frac_max=fn_pair_frac_max)
