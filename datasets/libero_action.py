"""Aligned LIBERO observations, actions, task ids, and robot state."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterator

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from datasets.base import as_hw
from datasets.libero import LiberoMixin


LOGGER = logging.getLogger(__name__)


LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
)

PHASE_NAMES = ("grasp", "release", "precision", "terminal")
PHASE_TO_ID = {name: index + 1 for index, name in enumerate(PHASE_NAMES)}


def _event_window_starts(
    events: np.ndarray,
    first: int,
    last: int,
    pre_event_steps: int,
    post_event_steps: int,
) -> list[int]:
    """Return legal action-window starts surrounding one or more events."""
    starts: set[int] = set()
    for event in np.asarray(events, dtype=np.int64).reshape(-1):
        lo = max(first, int(event) - pre_event_steps)
        hi = min(last, int(event) + post_event_steps)
        if lo <= hi:
            starts.update(range(lo, hi + 1))
    return sorted(starts)


def build_phase_start_candidates(
    actions: np.ndarray,
    first: int,
    last: int,
    *,
    action_horizon: int,
    pre_event_steps: int = 8,
    post_event_steps: int = 2,
    terminal_window_steps: int = 24,
) -> dict[str, list[int]]:
    """Build phase indices from a successful LIBERO trajectory."""
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 2:
        raise ValueError(f"expected actions [T,A>=2], got {actions.shape}")
    if not 0 <= first <= last < len(actions):
        raise ValueError(
            f"invalid legal range [{first}, {last}] for {len(actions)} actions"
        )

    gripper = actions[:, -1]
    closed = gripper > 0
    grasp_events = np.flatnonzero((~closed[:-1]) & closed[1:]) + 1
    release_events = np.flatnonzero(closed[:-1] & (~closed[1:])) + 1

    pose = actions[:, :-1]
    pose_norm = np.linalg.norm(pose, axis=1)
    correction = np.zeros(len(actions), dtype=np.float32)
    if len(actions) > 1:
        correction[1:] = np.linalg.norm(np.diff(pose, axis=0), axis=1)
    legal_indices = np.arange(first, last + 1, dtype=np.int64)
    closed_legal = legal_indices[closed[legal_indices]]
    precision_events = np.empty(0, dtype=np.int64)
    if len(closed_legal):
        low_motion = pose_norm[closed_legal] <= np.quantile(
            pose_norm[closed_legal], 0.40
        )
        high_correction = correction[closed_legal] >= np.quantile(
            correction[closed_legal], 0.75
        )
        raw = closed_legal[low_motion & high_correction]
        if len(raw):
            score = correction[raw] / (pose_norm[raw] + 1e-4)
            ordered = raw[np.argsort(score)[::-1]]
            selected: list[int] = []
            for index in ordered:
                if all(abs(int(index) - other) >= 4 for other in selected):
                    selected.append(int(index))
                    if len(selected) == 6:
                        break
            precision_events = np.asarray(sorted(selected), dtype=np.int64)

    terminal_span = max(action_horizon, terminal_window_steps)
    terminal_lo = max(first, last - terminal_span + 1)
    return {
        "grasp": _event_window_starts(
            grasp_events, first, last, pre_event_steps, post_event_steps
        ),
        "release": _event_window_starts(
            release_events, first, last, pre_event_steps, post_event_steps
        ),
        "precision": _event_window_starts(
            precision_events, first, last, pre_event_steps, post_event_steps
        ),
        "terminal": list(range(terminal_lo, last + 1)),
    }


def _weighted_choice(rng: random.Random, items: list[str], weights: list[float]) -> str:
    total = float(sum(weights))
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights, strict=True):
        cumulative += float(weight)
        if threshold < cumulative:
            return item
    return items[-1]


class _LiberoActionMixin(LiberoMixin):
    def _init_action_dataset(
        self,
        root: str | None,
        suites: tuple[str, ...],
        cameras: tuple[str, ...],
        camera_layout: str,
        frame_size: int | tuple[int, int],
        context_frames: int,
        context_frame_stride: int,
        world_prediction_steps: int,
        world_frame_stride: int,
        action_horizon: int,
        action_history_steps: int,
        fps: float,
        rollout_roots: tuple[str, ...] = (),
        rollout_success_filter: str = "all",
        expert_demo_whitelist_path: str | None = None,
    ) -> None:
        if min(
            context_frames,
            context_frame_stride,
            world_prediction_steps,
            world_frame_stride,
            action_horizon,
        ) <= 0 or fps <= 0:
            raise ValueError("frame counts, strides, action_horizon, and fps must be positive")
        if action_history_steps < 0:
            raise ValueError("action_history_steps must be non-negative")
        self.root = self._resolve_root(root)
        self.suites = tuple(suites)
        self.cameras = tuple(cameras)
        self.camera_layout = camera_layout
        self.frame_size = as_hw(frame_size)
        self.context_frames = int(context_frames)
        self.context_frame_stride = int(context_frame_stride)
        self.world_prediction_steps = int(world_prediction_steps)
        self.world_frame_stride = int(world_frame_stride)
        self.action_horizon = int(action_horizon)
        self.action_history_steps = int(action_history_steps)
        self.fps = float(fps)
        self._validate_camera_args()
        self.expert_samples = self._load_demo_samples(self.root, self.suites)
        unfiltered_expert_count = len(self.expert_samples)
        self.expert_samples = self._filter_demo_samples_by_whitelist(
            self.expert_samples, self.root, expert_demo_whitelist_path
        )
        if expert_demo_whitelist_path is not None:
            LOGGER.info(
                "Strict replay whitelist retained %d/%d LIBERO expert demos from %s",
                len(self.expert_samples),
                unfiltered_expert_count,
                expert_demo_whitelist_path,
            )
        self.rollout_success_filter = str(rollout_success_filter)
        if self.rollout_success_filter not in {"all", "success", "failure"}:
            raise ValueError(
                "rollout_success_filter must be 'all', 'success', or 'failure'"
            )
        self.rollout_samples = self._load_rollout_samples(
            tuple(Path(item) for item in rollout_roots),
            self.suites,
            self.rollout_success_filter,
        )
        self.demo_samples = self.expert_samples + self.rollout_samples
        self.samples_by_suite = self._group_samples_by_suite(self.demo_samples, self.suites)
        self.samples_by_task = self._group_samples_by_task(self.demo_samples)
        expert_tasks = set(self._group_samples_by_task(self.expert_samples))
        rollout_tasks = set(self._group_samples_by_task(self.rollout_samples))
        self.task_names = tuple(sorted(expert_tasks))
        self.task_to_id = self._build_global_task_vocabulary(self.root)
        unknown_rollout_tasks = sorted(rollout_tasks - set(self.task_to_id))
        if unknown_rollout_tasks:
            raise ValueError(
                f"rollout task names are absent from LIBERO demos: {unknown_rollout_tasks}"
            )
        if not self.task_names:
            raise FileNotFoundError(f"No LIBERO tasks found for suites {self.suites} under {self.root}")
        for suite in self.suites:
            suite_task_names = {
                self._task_name_for_sample(sample)
                for sample in self.samples_by_suite[suite]
            }
            if suite == "libero_90" and len(suite_task_names) != 90:
                raise ValueError(
                    "Expected 90 tasks inside the LIBERO-90 suite, but got "
                    f"{len(suite_task_names)} tasks. "
                    "Check HDF5 filenames and suite selection."
                )
        LOGGER.info(
            "LIBERO action dataset loaded %d expert + %d rollout samples "
            "from %d tasks across suites=%s",
            len(self.expert_samples), len(self.rollout_samples),
            len(self.task_names), self.suites,
        )

    @staticmethod
    def _load_rollout_samples(
        roots: tuple[Path, ...],
        suites: tuple[str, ...],
        success_filter: str,
    ) -> list[tuple[str, str]]:
        samples: list[tuple[str, str]] = []
        for root in roots:
            for suite in suites:
                suite_dir = root / suite
                for path in sorted(suite_dir.glob("*_demo.hdf5")):
                    with h5py.File(path, "r") as h5:
                        for demo_name in sorted(h5["data"]):
                            demo = h5["data"][demo_name]
                            if str(demo.attrs.get("source", "")) != "policy_rollout":
                                continue
                            success = bool(demo.attrs.get("success", False))
                            if success_filter == "success" and not success:
                                continue
                            if success_filter == "failure" and success:
                                continue
                            samples.append((str(path), demo_name))
        return samples


    @staticmethod
    def _task_name_for_sample(sample: tuple[str, str]) -> str:
        return Path(sample[0]).stem.removesuffix("_demo")

    @classmethod
    def _group_samples_by_task(
        cls, samples: list[tuple[str, str]]
    ) -> dict[str, list[tuple[str, str]]]:
        grouped: dict[str, list[tuple[str, str]]] = {}
        for sample in samples:
            grouped.setdefault(cls._task_name_for_sample(sample), []).append(sample)
        return grouped

    @staticmethod
    def _distributed_rank() -> int:
        for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
            value = os.environ.get(key)
            if value is not None:
                try:
                    return int(value)
                except ValueError:
                    continue
        return 0

    @staticmethod
    def _build_global_task_vocabulary(root: Path) -> dict[str, int]:
        # Scan every installed suite so task ids remain stable when the YAML
        # changes train/validation suite subsets.
        task_names = sorted(
            {
                path.stem.removesuffix("_demo")
                for suite in LIBERO_SUITES
                for path in (root / suite).glob("*.hdf5")
            }
        )
        if not task_names:
            raise FileNotFoundError(f"No LIBERO tasks found under {root}")
        return {name: index for index, name in enumerate(task_names)}

    @property
    def num_tasks(self) -> int:
        return len(self.task_to_id)

    def _valid_start_range(self, num_steps: int) -> tuple[int, int]:
        first = (self.context_frames - 1) * self.context_frame_stride
        # Future world timestamps are predictor queries, not frames read here.
        last_action = num_steps - self.action_horizon
        last_future = num_steps - 1 - self.world_prediction_steps * self.world_frame_stride
        last = min(last_action, last_future)
        if last < first:
            raise ValueError(
                f"demo length {num_steps} is too short for context={self.context_frames}, "
                f"stride={self.context_frame_stride}, horizon={self.action_horizon}"
            )
        return first, last

    @staticmethod
    def _instruction(h5: h5py.File, path: Path) -> str:
        raw = h5["data"].attrs.get("problem_info")
        if raw:
            try:
                return str(json.loads(raw).get("language_instruction") or path.stem)
            except (TypeError, json.JSONDecodeError):
                pass
        return path.stem.removesuffix("_demo").replace("_", " ")

    def _resize_frames(self, frames: np.ndarray) -> torch.Tensor:
        out_h, out_w = self.frame_size
        if frames.ndim == 4:
            frames = np.stack([cv2.resize(frame, (out_w, out_h)) for frame in frames])
            if frames.dtype != np.uint8:
                frames = np.clip(frames, 0, 255).astype(np.uint8)
            return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        if frames.ndim == 5:
            # Multi-view layout from LiberoMixin: [T,V,H,W,3] -> [T,V,C,H,W].
            resized = np.stack([
                np.stack([cv2.resize(view, (out_w, out_h)) for view in timestep], axis=0)
                for timestep in frames
            ], axis=0)
            if resized.dtype != np.uint8:
                resized = np.clip(resized, 0, 255).astype(np.uint8)
            return torch.from_numpy(np.ascontiguousarray(resized)).permute(0, 1, 4, 2, 3)
        raise ValueError(f"expected frames [T,H,W,3] or [T,V,H,W,3], got {frames.shape}")

    def _read_sample(self, sample_index: int, action_start: int) -> dict[str, Any]:
        path_string, demo_name = self.demo_samples[sample_index]
        path = Path(path_string)
        with h5py.File(path, "r") as h5:
            demo = h5["data"][demo_name]
            obs = demo["obs"]
            num_steps = len(demo["actions"])
            first, last = self._valid_start_range(num_steps)
            action_start = min(max(int(action_start), first), last)
            frame_indices = [
                action_start - (self.context_frames - 1 - i) * self.context_frame_stride
                for i in range(self.context_frames)
            ]
            frames = self._read_camera_frames(obs, frame_indices)
            future_frame_indices = [
                action_start + self.world_frame_stride * i
                for i in range(1, self.world_prediction_steps + 1)
            ]
            future_frames = self._read_camera_frames(obs, future_frame_indices)
            actions_dataset = demo["actions"]
            actions = np.asarray(
                actions_dataset[action_start : action_start + self.action_horizon],
                dtype=np.float32,
            )
            action_dim = int(actions_dataset.shape[-1])
            action_history = np.zeros(
                (self.action_history_steps, action_dim), dtype=np.float32
            )
            action_history_mask = np.zeros(self.action_history_steps, dtype=np.bool_)
            if self.action_history_steps > 0:
                history_start = max(0, action_start - self.action_history_steps)
                valid_history = np.asarray(
                    actions_dataset[history_start:action_start], dtype=np.float32
                )
                valid_count = len(valid_history)
                if valid_count:
                    # Right alignment preserves physical relative times -k,...,-1.
                    action_history[-valid_count:] = valid_history
                    action_history_mask[-valid_count:] = True
            if "robot_states" in demo:
                proprio = np.asarray(demo["robot_states"][action_start], dtype=np.float32)
            else:
                proprio = np.concatenate(
                    [
                        np.asarray(obs["ee_states"][action_start], dtype=np.float32),
                        np.asarray(obs["gripper_states"][action_start], dtype=np.float32)[:1],
                    ]
                )
            instruction = self._instruction(h5, path)

        base_index = frame_indices[0]
        observed_timestamps = (
            torch.tensor(frame_indices, dtype=torch.float32) - float(base_index)
        ) / self.fps
        future_indices = torch.tensor(future_frame_indices, dtype=torch.float32)
        future_timestamps = (future_indices - float(base_index)) / self.fps
        task_name = path.stem.removesuffix("_demo")
        return {
            "video": self._resize_frames(frames),
            "future_video": self._resize_frames(future_frames),
            "timestamps": observed_timestamps,
            "future_timestamps": future_timestamps,
            "action": torch.from_numpy(actions),
            "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "action_history": torch.from_numpy(action_history),
            "action_history_mask": torch.from_numpy(action_history_mask),
            "proprio": torch.from_numpy(proprio),
            "task_id": torch.tensor(self.task_to_id[task_name], dtype=torch.long),
            "task_name": task_name,
            "instruction": instruction,
        }


class LiberoActionTrain(_LiberoActionMixin, IterableDataset):
    """Infinite random-window LIBERO action dataset."""

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
        suite_sampling: str = "proportional",
        task_sampling: str = "proportional",
        base_all_tasks_probability: float = 1.0,
        hard_tasks_probability: float = 0.0,
        hard_task_names: tuple[str, ...] = (),
        hard_task_phase_sampling_probability: float = 0.6,
        phase_sampling_weights: dict[str, float] | None = None,
        phase_pre_event_steps: int = 8,
        phase_post_event_steps: int = 2,
        phase_terminal_window_steps: int = 24,
        rollout_roots: tuple[str, ...] = (),
        rollout_probability: float = 0.0,
        rollout_success_filter: str = "all",
        expert_demo_whitelist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.seed = int(seed)
        self.suite_sampling = str(suite_sampling)
        self.task_sampling = str(task_sampling)
        self.base_all_tasks_probability = float(base_all_tasks_probability)
        self.hard_tasks_probability = float(hard_tasks_probability)
        self.hard_task_names = tuple(str(name) for name in hard_task_names)
        self.hard_task_phase_sampling_probability = float(
            hard_task_phase_sampling_probability
        )
        self.phase_sampling_weights = {
            name: float((phase_sampling_weights or {}).get(name, default))
            for name, default in zip(PHASE_NAMES, (0.2, 0.3, 0.3, 0.2), strict=True)
        }
        self.phase_pre_event_steps = int(phase_pre_event_steps)
        self.phase_post_event_steps = int(phase_post_event_steps)
        self.phase_terminal_window_steps = int(phase_terminal_window_steps)
        self.rollout_probability = float(rollout_probability)
        if not 0.0 <= self.rollout_probability <= 1.0:
            raise ValueError("rollout_probability must be in [0,1]")
        if self.suite_sampling not in {"proportional", "balanced"}:
            raise ValueError(
                f"suite_sampling must be 'proportional' or 'balanced', got {self.suite_sampling!r}"
            )
        if self.task_sampling not in {"proportional", "balanced"}:
            raise ValueError(
                f"task_sampling must be 'proportional' or 'balanced', got {self.task_sampling!r}"
            )
        self._init_action_dataset(
            root, suites, cameras, camera_layout, frame_size, context_frames,
            context_frame_stride, world_prediction_steps, world_frame_stride,
            action_horizon, action_history_steps, fps,
            rollout_roots, rollout_success_filter, expert_demo_whitelist_path,
        )
        if self.rollout_probability > 0 and not self.rollout_samples:
            raise FileNotFoundError(
                "rollout_probability is positive but rollout_roots contain no matching samples"
            )
        self._expert_samples_by_task = self._group_samples_by_task(self.expert_samples)
        self._expert_samples_by_suite = self._group_samples_by_suite(self.expert_samples, self.suites)
        self._rollout_samples_by_task = self._group_samples_by_task(self.rollout_samples)
        self._validate_phase_sampling()
        self._sample_to_index = {
            sample: index for index, sample in enumerate(self.demo_samples)
        }
        self._phase_candidate_cache: dict[
            tuple[str, str], dict[str, list[int]]
        ] = {}

    def _validate_phase_sampling(self) -> None:
        probabilities = (
            self.base_all_tasks_probability,
            self.hard_tasks_probability,
            self.hard_task_phase_sampling_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError(
                f"sampling probabilities must be in [0,1], got {probabilities}"
            )
        if not np.isclose(
            self.base_all_tasks_probability + self.hard_tasks_probability, 1.0
        ):
            raise ValueError(
                "base_all_tasks_probability + hard_tasks_probability must equal 1"
            )
        if self.phase_pre_event_steps < 0 or self.phase_post_event_steps < 0:
            raise ValueError("phase event offsets must be non-negative")
        if self.phase_terminal_window_steps <= 0:
            raise ValueError("phase_terminal_window_steps must be positive")
        if any(value < 0 for value in self.phase_sampling_weights.values()):
            raise ValueError("phase sampling weights must be non-negative")
        if sum(self.phase_sampling_weights.values()) <= 0:
            raise ValueError("at least one phase sampling weight must be positive")
        if self.hard_tasks_probability > 0:
            if not self.hard_task_names:
                raise ValueError("hard_task_names is required when hard sampling is enabled")
            missing = sorted(set(self.hard_task_names) - set(self.task_names))
            if missing:
                raise ValueError(
                    "hard-task names do not exactly match loaded HDF5 tasks: "
                    + ", ".join(missing)
                )
        LOGGER.info(
            "LIBERO hierarchical sampling base=%.3f hard=%.3f hard_phase=%.3f "
            "hard_tasks=%d phase_weights=%s",
            self.base_all_tasks_probability,
            self.hard_tasks_probability,
            self.hard_task_phase_sampling_probability,
            len(self.hard_task_names),
            self.phase_sampling_weights,
        )

    def _choose_task_and_sample(
        self, rng: random.Random
    ) -> tuple[int, str, str, bool, bool]:
        """Choose source -> task -> trajectory; report hard/rollout branches."""
        rollout_branch = bool(
            self.rollout_samples and rng.random() < self.rollout_probability
        )
        if rollout_branch:
            available_tasks = tuple(sorted(self._rollout_samples_by_task))
            task_name = available_tasks[rng.randrange(len(available_tasks))]
            sample = rng.choice(self._rollout_samples_by_task[task_name])
            return self._sample_to_index[sample], sample[0], sample[1], False, True
        hard_branch = (
            self.hard_tasks_probability > 0
            and rng.random() >= self.base_all_tasks_probability
        )
        if hard_branch:
            task_name = self.hard_task_names[rng.randrange(len(self.hard_task_names))]
            sample = rng.choice(self._expert_samples_by_task[task_name])
        elif self.task_sampling == "balanced" or self.hard_tasks_probability > 0:
            expert_tasks = tuple(sorted(self._expert_samples_by_task))
            task_name = expert_tasks[rng.randrange(len(expert_tasks))]
            sample = rng.choice(self._expert_samples_by_task[task_name])
        elif self.suite_sampling == "balanced":
            suite = self.suites[rng.randrange(len(self.suites))]
            sample = rng.choice(self._expert_samples_by_suite[suite])
        else:
            sample = self.expert_samples[rng.randrange(len(self.expert_samples))]
        return self._sample_to_index[sample], sample[0], sample[1], hard_branch, False

    def _phase_candidates(
        self,
        path: str,
        demo_name: str,
        actions: np.ndarray,
        first: int,
        last: int,
    ) -> dict[str, list[int]]:
        key = (path, demo_name)
        cached = self._phase_candidate_cache.get(key)
        if cached is None:
            cached = build_phase_start_candidates(
                actions,
                first,
                last,
                action_horizon=self.action_horizon,
                pre_event_steps=self.phase_pre_event_steps,
                post_event_steps=self.phase_post_event_steps,
                terminal_window_steps=self.phase_terminal_window_steps,
            )
            self._phase_candidate_cache[key] = cached
        return cached

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rank = self._distributed_rank()
        # IterableDataset workers are replicated on each DDP rank.  Offset the
        # RNG by rank and worker so different GPUs/workers do not redundantly
        # sample the same task/demo/window stream.
        rng = random.Random(self.seed + 1_000_003 * rank + 1009 * worker_id)
        while True:
            sample_index, path, demo_name, hard_branch, rollout_branch = self._choose_task_and_sample(rng)
            try:
                with h5py.File(path, "r") as h5:
                    actions_dataset = h5["data"][demo_name]["actions"]
                    num_steps = len(actions_dataset)
                    phase_requested = (
                        hard_branch
                        and rng.random() < self.hard_task_phase_sampling_probability
                    )
                    actions = (
                        np.asarray(actions_dataset, dtype=np.float32)
                        if phase_requested
                        else None
                    )
                first, last = self._valid_start_range(num_steps)
                phase_name = "uniform"
                if phase_requested and actions is not None:
                    candidates = self._phase_candidates(
                        path, demo_name, actions, first, last
                    )
                    available = [
                        name for name in PHASE_NAMES
                        if candidates[name] and self.phase_sampling_weights[name] > 0
                    ]
                    if available:
                        phase_name = _weighted_choice(
                            rng,
                            available,
                            [self.phase_sampling_weights[name] for name in available],
                        )
                        action_start = rng.choice(candidates[phase_name])
                    else:
                        action_start = rng.randint(first, last)
                else:
                    action_start = rng.randint(first, last)
                sample = self._read_sample(sample_index, action_start)
                task_name = self._task_name_for_sample((path, demo_name))
                sample.update(
                    {
                        "sampling_base_branch": torch.tensor(not hard_branch),
                        "sampling_hard_branch": torch.tensor(hard_branch),
                        "sampling_task_is_hard": torch.tensor(
                            task_name in self.hard_task_names
                        ),
                        "sampling_is_phase": torch.tensor(phase_name != "uniform"),
                        "sampling_phase_id": torch.tensor(
                            PHASE_TO_ID.get(phase_name, 0), dtype=torch.long
                        ),
                        "sampling_is_rollout": torch.tensor(rollout_branch),
                    }
                )
                yield sample
            except (OSError, KeyError, ValueError):
                continue


class LiberoActionVal(_LiberoActionMixin, Dataset):
    """One deterministic center window per LIBERO demonstration."""

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
        max_samples: int | None = 256,
        seed: int = 0,
        expert_demo_whitelist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.seed = int(seed)
        self._init_action_dataset(
            root, suites, cameras, camera_layout, frame_size, context_frames,
            context_frame_stride, world_prediction_steps, world_frame_stride,
            action_horizon, action_history_steps, fps,
            (), "all", expert_demo_whitelist_path,
        )
        if max_samples is not None:
            self.demo_samples = self.demo_samples[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.demo_samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, demo_name = self.demo_samples[index]
        with h5py.File(path, "r") as h5:
            num_steps = len(h5["data"][demo_name]["actions"])
        first, last = self._valid_start_range(num_steps)
        return self._read_sample(index, (first + last) // 2)
