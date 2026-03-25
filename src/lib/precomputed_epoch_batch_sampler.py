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

    def __init__(self, precomputed_path: Path | str) -> None:
        path = Path(precomputed_path)
        payload = load_precomputed_batches_payload(path)
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError(f"Неизвестный format_version в {path}: {payload.get('format_version')}")

        self.batches_by_epoch: List[List[List[int]]] = payload["batches_by_epoch"]
        self.stored_size = payload.get("dataset_size")
        file_args = payload.get("args") or {}
        self.file_bs = int(file_args.get("batch_size", 0))
        self.file_drop_last = bool(file_args.get("drop_last", False))

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
        return PrecomputedEpochBatchSampler(
            dataset,
            batch_size=int(batch_size),
            drop_last=bool(drop_last),
            batches_by_epoch=self.batches_by_epoch,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )


def create_precomputed_batch_sampler_factory(precomputed_path: Path | str) -> PrecomputedEpochBatchSamplerFactory:
    """Обёртка для совместимости."""
    return PrecomputedEpochBatchSamplerFactory(precomputed_path)
