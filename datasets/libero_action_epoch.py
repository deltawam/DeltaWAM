"""Finite LIBERO action dataset for epoch-based shuffle-buffer training."""

from __future__ import annotations

import logging
from typing import Any

import h5py
import torch
from torch.utils.data import Dataset

from datasets.libero_action import _LiberoActionMixin


LOGGER = logging.getLogger(__name__)


class LiberoActionEpochTrain(_LiberoActionMixin, Dataset):
    """Index every legal (demo, action_start) window exactly once per epoch."""

    def __init__(
        self,
        root: str | None = None,
        suites: tuple[str, ...] = ("libero_spatial",),
        cameras: tuple[str, ...] = ("agentview_rgb",),
        camera_layout: str = "first",
        frame_size: int | tuple[int, int] = 512,
        context_frames: int = 4,
        context_frame_stride: int = 2,
        world_prediction_steps: int = 3,
        world_frame_stride: int = 2,
        action_horizon: int = 16,
        action_history_steps: int = 0,
        fps: float = 20.0,
        seed: int = 0,
        shuffle_buffer_size: int = 100_000,
        rollout_roots: tuple[str, ...] = (),
        rollout_success_filter: str = "all",
    ) -> None:
        super().__init__()
        self.seed = int(seed)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        if self.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        self._init_action_dataset(
            root,
            suites,
            cameras,
            camera_layout,
            frame_size,
            context_frames,
            context_frame_stride,
            world_prediction_steps,
            world_frame_stride,
            action_horizon,
            action_history_steps,
            fps,
            rollout_roots,
            rollout_success_filter,
        )

        self.window_index: list[tuple[int, int]] = []
        for sample_index, (path, demo_name) in enumerate(self.demo_samples):
            with h5py.File(path, "r") as h5:
                num_steps = len(h5["data"][demo_name]["actions"])
            try:
                first, last = self._valid_start_range(num_steps)
            except ValueError:
                LOGGER.warning(
                    "Skipping short LIBERO demo in epoch dataset: %s/%s",
                    path,
                    demo_name,
                )
                continue
            self.window_index.extend(
                (sample_index, action_start)
                for action_start in range(first, last + 1)
            )

        if not self.window_index:
            raise FileNotFoundError(
                f"No legal LIBERO action windows for suites {self.suites}"
            )
        LOGGER.info(
            "LIBERO epoch dataset indexed %d legal windows from %d demos "
            "across suites=%s with shuffle_buffer_size=%d",
            len(self.window_index),
            len(self.demo_samples),
            self.suites,
            self.shuffle_buffer_size,
        )

    def __len__(self) -> int:
        return len(self.window_index)


    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, action_start = self.window_index[index]
        sample = self._read_sample(sample_index, action_start)
        sample.update(
            {
                "sampling_base_branch": torch.tensor(True),
                "sampling_hard_branch": torch.tensor(False),
                "sampling_task_is_hard": torch.tensor(False),
                "sampling_is_phase": torch.tensor(False),
                "sampling_phase_id": torch.tensor(0, dtype=torch.long),
                "sampling_is_rollout": torch.tensor(
                    sample_index >= len(self.expert_samples)
                ),
            }
        )
        return sample
