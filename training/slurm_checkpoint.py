from __future__ import annotations

import logging
import os
import signal
from pathlib import Path
from typing import Any

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback


class SlurmPreemptionCheckpoint(Callback):
    """Save a full, resumable last.ckpt on a Slurm pre-timeout signal."""

    def __init__(self, signal_number: int = signal.SIGUSR2) -> None:
        self.signal_number = signal_number
        self._signal_received = False
        self._saved = False
        self._previous_handler: Any = None

    def _handle_signal(self, signum: int, frame: Any) -> None:
        del frame
        self._signal_received = True
        logging.warning(
            "Received pre-timeout signal %s; checkpoint will be saved at the "
            "next safe batch boundary.",
            signum,
        )

    def on_fit_start(self, trainer: Trainer, pl_module: Any) -> None:
        del trainer, pl_module
        if not os.environ.get("SLURM_JOB_ID"):
            return
        self._previous_handler = signal.getsignal(self.signal_number)
        signal.signal(self.signal_number, self._handle_signal)
        logging.info(
            "Slurm pre-timeout checkpoint handler registered for signal %s.",
            self.signal_number,
        )

    def _any_rank_received_signal(self, pl_module: Any) -> bool:
        if not os.environ.get("SLURM_JOB_ID"):
            return False
        received = torch.tensor(
            int(self._signal_received), device=pl_module.device, dtype=torch.int32
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(received, op=torch.distributed.ReduceOp.SUM)
        return bool(received.item())

    @staticmethod
    def _last_checkpoint_path(trainer: Trainer) -> Path:
        for callback in trainer.checkpoint_callbacks:
            if getattr(callback, "save_last", False) and callback.dirpath:
                return Path(callback.dirpath) / "last.ckpt"
        return Path(trainer.default_root_dir) / "checkpoints" / "last.ckpt"

    def _save_if_requested(self, trainer: Trainer, pl_module: Any) -> None:
        if self._saved or not self._any_rank_received_signal(pl_module):
            return

        last_path = self._last_checkpoint_path(trainer)
        temporary_path = last_path.with_name(f".{last_path.name}.slurm-tmp")
        if trainer.is_global_zero:
            last_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.strategy.barrier("slurm-preemption-checkpoint-dir")

        logging.warning(
            "Slurm allocation is nearing its time limit; saving full checkpoint "
            "at global_step=%d to %s.",
            trainer.global_step,
            last_path,
        )
        trainer.save_checkpoint(str(temporary_path), weights_only=False)
        trainer.strategy.barrier("slurm-preemption-checkpoint-written")
        if trainer.is_global_zero:
            os.replace(temporary_path, last_path)
        trainer.strategy.barrier("slurm-preemption-checkpoint-published")

        self._saved = True
        trainer.should_stop = True
        logging.warning(
            "Pre-timeout checkpoint complete at global_step=%d; stopping cleanly.",
            trainer.global_step,
        )

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        del outputs, batch, batch_idx
        self._save_if_requested(trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, batch, batch_idx, dataloader_idx
        if not trainer.sanity_checking:
            self._save_if_requested(trainer, pl_module)

    def on_fit_end(self, trainer: Trainer, pl_module: Any) -> None:
        del trainer, pl_module
        if self._previous_handler is not None:
            signal.signal(self.signal_number, self._previous_handler)
            self._previous_handler = None
