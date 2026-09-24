"""Lightning datamodule for dict-valued action samples."""

from __future__ import annotations

from typing import Any

from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from datasets.module import _load_cls
from datasets.shuffle_buffer_sampler import ShuffleBufferDistributedSampler


class ActionDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_cfg: dict[str, Any],
        val_dataset_cfg: dict[str, Any] | None = None,
        batch_size: int = 2,
        val_batch_size: int = 2,
        num_workers: int = 4,
        frame_size: int | tuple[int, int] = 512,
        prefetch_factor: int | None = 2,
    ) -> None:
        super().__init__()
        self.train_dataset_cfg = train_dataset_cfg
        self.val_dataset_cfg = val_dataset_cfg
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.frame_size = frame_size
        self.prefetch_factor = prefetch_factor

    @staticmethod
    def _build(cfg: dict[str, Any], frame_size):
        cls = _load_cls(cfg["class_path"])
        return cls(**cfg.get("init_args", {}), frame_size=frame_size)

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.train_dataset = self._build(self.train_dataset_cfg, self.frame_size)
        self.val_dataset = (
            self._build(self.val_dataset_cfg, self.frame_size)
            if self.val_dataset_cfg is not None
            else None
        )

    def _loader(
        self, dataset, batch_size: int, *, training: bool = False
    ) -> DataLoader:
        sampler = None
        shuffle = False
        if training and not isinstance(dataset, IterableDataset):
            buffer_size = getattr(dataset, "shuffle_buffer_size", None)
            if buffer_size is not None:
                sampler = ShuffleBufferDistributedSampler(
                    dataset,
                    buffer_size=int(buffer_size),
                    seed=int(getattr(dataset, "seed", 0)),
                )
            else:
                shuffle = True
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context="fork" if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self.batch_size, training=True)

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        return self._loader(self.val_dataset, self.val_batch_size)
