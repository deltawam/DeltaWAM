import json
import os
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch

from training.auto_prepare import ensure_libero_suites

from datasets.base import (
    TrainDataset,
    ValDataset,
    compute_resize_sizes,
    sample_frame_indices,
)


class LiberoMixin:
    def _resolve_root(self, root: str | None) -> Path:
        default_root = Path(__file__).resolve().parents[1] / "data" / "libero"
        resolved = Path(root or os.environ.get("LIBERO_ROOT", default_root))
        return resolved

    def _load_demo_samples(
        self, root: Path, suites: tuple[str, ...]
    ) -> list[tuple[str, str]]:
        ensure_libero_suites(root, suites)
        files = []
        for suite in suites:
            suite_dir = root / suite
            files.extend(sorted(suite_dir.glob("*.hdf5")))
        if not files:
            raise FileNotFoundError(
                f"No LIBERO hdf5 files found under {root} for suites {suites}"
            )

        samples = []
        for path in files:
            with h5py.File(path, "r") as h5:
                demos = sorted(
                    h5["data"].keys(),
                    key=lambda name: (0, int(name.rsplit("_", 1)[-1]))
                    if name.rsplit("_", 1)[-1].isdigit()
                    else (1, name),
                )
                samples.extend((str(path), demo) for demo in demos)
        return samples

    @staticmethod
    def _filter_demo_samples_by_whitelist(
        samples: list[tuple[str, str]],
        root: Path,
        whitelist_path: str | None,
    ) -> list[tuple[str, str]]:
        if whitelist_path is None:
            return samples

        path = Path(whitelist_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LIBERO expert demo whitelist not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_criterion = (
            "terminal env.check_success() after exact action replay"
        )
        if payload.get("criterion") != expected_criterion:
            raise ValueError(
                "Refusing a non-strict LIBERO whitelist: expected criterion "
                f"{expected_criterion!r}, got {payload.get('criterion')!r}"
            )
        demos = payload.get("demos")
        if not isinstance(demos, dict):
            raise ValueError("LIBERO whitelist 'demos' must be an object")

        allowed = {
            (Path(hdf5_name).as_posix(), str(demo_name))
            for hdf5_name, demo_names in demos.items()
            for demo_name in demo_names
        }
        root = root.expanduser().resolve()
        filtered = []
        for hdf5_name, demo_name in samples:
            hdf5_path = Path(hdf5_name).expanduser().resolve()
            try:
                relative_name = hdf5_path.relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError(
                    f"LIBERO sample {hdf5_path} is outside dataset root {root}"
                ) from error
            if (relative_name, demo_name) in allowed:
                filtered.append((hdf5_name, demo_name))

        if not filtered:
            raise ValueError(
                f"Strict LIBERO whitelist {path} retained zero demos under {root}"
            )
        return filtered

    @staticmethod
    def _suite_name_for_sample(sample: tuple[str, str]) -> str:
        return Path(sample[0]).parent.name

    @staticmethod
    def _group_samples_by_suite(
        samples: list[tuple[str, str]], suites: tuple[str, ...]
    ) -> dict[str, list[tuple[str, str]]]:
        grouped = {suite: [] for suite in suites}
        for sample in samples:
            suite = LiberoMixin._suite_name_for_sample(sample)
            if suite in grouped:
                grouped[suite].append(sample)
        missing = [suite for suite, items in grouped.items() if not items]
        if missing:
            raise FileNotFoundError(f"No LIBERO samples found for suites {missing}")
        return grouped


    @staticmethod
    def _instruction(h5: h5py.File, path: Path) -> str:
        raw = h5["data"].attrs.get("problem_info")
        if raw:
            try:
                return str(json.loads(raw).get("language_instruction") or path.stem)
            except (TypeError, json.JSONDecodeError):
                pass
        return path.stem.removesuffix("_demo").replace("_", " ")

    def _read_camera_frames(self, obs: Any, frame_indices: list[int]) -> np.ndarray:
        frames = []
        for camera in self.cameras:
            if camera not in obs:
                raise KeyError(f"Camera {camera!r} not found in LIBERO obs keys {list(obs)}")
            frames.append(np.asarray(obs[camera][frame_indices]))

        if self.camera_layout == "multi_view":
            return np.stack(frames, axis=1)
        if self.camera_layout == "first" or len(frames) == 1:
            return frames[0]
        axis = 2 if self.camera_layout == "horizontal" else 1
        return np.concatenate(frames, axis=axis)

    def _validate_camera_args(self) -> None:
        if self.camera_layout not in {"first", "horizontal", "vertical", "multi_view"}:
            raise ValueError(f"Unsupported camera_layout: {self.camera_layout}")
        if not self.cameras:
            raise ValueError("At least one camera must be specified")


class LiberoTrain(LiberoMixin, TrainDataset):
    def __init__(
        self,
        root: str | None = None,
        suites: tuple[str, ...] = (
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_90",
            "libero_10",
        ),
        cameras: tuple[str, ...] = ("agentview_rgb",),
        camera_layout: str = "first",
        num_frames: int = 8,
        frame_size: int = 256,
        fps: float = 20.0,
        time_stride_range: tuple[float, float] = (1 / 20, 1 / 3),
        horizontal_flip: bool = False,
        ratio_jitter: float = 4 / 3,
        scale: tuple[float, float] = (0.8, 1.0),
        suite_sampling: str = "proportional",
    ):
        super().__init__(
            root=root,
            suites=suites,
            cameras=cameras,
            camera_layout=camera_layout,
            num_frames=num_frames,
            frame_size=frame_size,
            fps=fps,
            time_stride_range=time_stride_range,
            horizontal_flip=horizontal_flip,
            ratio_jitter=ratio_jitter,
            scale=scale,
        )
        self._validate_camera_args()
        self.root = self._resolve_root(root)
        self.samples = self._load_demo_samples(self.root, self.suites)
        self.suite_sampling = str(suite_sampling)
        if self.suite_sampling not in {"proportional", "balanced"}:
            raise ValueError(
                f"suite_sampling must be 'proportional' or 'balanced', got {self.suite_sampling!r}"
            )
        self.samples_by_suite = self._group_samples_by_suite(self.samples, self.suites)
        self._balanced_len = len(self.suites) * max(
            len(items) for items in self.samples_by_suite.values()
        )

    def _len(self) -> int:
        if self.suite_sampling == "balanced":
            return self._balanced_len
        return len(self.samples)

    def _resolve_training_sample(self, idx: int) -> tuple[str, str]:
        if self.suite_sampling == "proportional":
            return self.samples[idx % len(self.samples)]
        suite = self.suites[idx % len(self.suites)]
        suite_samples = self.samples_by_suite[suite]
        sample_offset = (idx // len(self.suites)) % len(suite_samples)
        return suite_samples[sample_offset]

    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, demo = self._resolve_training_sample(idx)
        with h5py.File(path, "r") as h5:
            obs = h5["data"][demo]["obs"]
            num_steps = len(obs[self.cameras[0]])
            timestamps = torch.arange(num_steps, dtype=torch.float32) / self.fps
            frame_indices, sampled_timestamps = sample_frame_indices(
                timestamps, self.num_frames, self.time_stride_range
            )
            frames = self._read_camera_frames(obs, frame_indices)
            instruction = self._instruction(h5, Path(path))

        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        if frames.shape[-1] != 3:
            raise ValueError(f"Expected channel-last RGB frames, got shape {frames.shape}")
        frames = np.ascontiguousarray(frames)
        out, sampled_timestamps = self._augment(frames, sampled_timestamps)
        return out, sampled_timestamps - sampled_timestamps[0], instruction


class LiberoVal(LiberoMixin, ValDataset):
    def __init__(
        self,
        root: str | None = None,
        suites: tuple[str, ...] = ("libero_10",),
        cameras: tuple[str, ...] = ("agentview_rgb",),
        camera_layout: str = "first",
        num_frames: int = 8,
        frame_size: int = 256,
        fps: float = 20.0,
        time_stride_seconds: float = 0.1,
        max_aspect_ratio: float = 2.0,
        max_samples: int | None = 256,
    ):
        self.root = self._resolve_root(root)
        self.suites = suites
        self.cameras = cameras
        self.camera_layout = camera_layout
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.fps = fps
        self.time_stride_seconds = time_stride_seconds
        self.max_aspect_ratio = max_aspect_ratio
        self.max_samples = max_samples
        self._validate_camera_args()
        self.samples = self._load_demo_samples(self.root, self.suites)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, None, int]:
        path, demo = self.samples[idx]
        with h5py.File(path, "r") as h5:
            obs = h5["data"][demo]["obs"]
            num_steps = len(obs[self.cameras[0]])
            timestamps = torch.arange(num_steps, dtype=torch.float32) / self.fps
            stride = max(1, round(self.time_stride_seconds * self.fps))
            total_span = (self.num_frames - 1) * stride
            start = max(0, num_steps // 2 - total_span // 2)
            start = min(start, num_steps - 1 - total_span)
            indices = [start + i * stride for i in range(self.num_frames)]
            frames = self._read_camera_frames(obs, indices)
            sampled_timestamps = timestamps[indices]
            instruction = self._instruction(h5, Path(path))

        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        if frames.ndim == 4:
            h, w = frames[0].shape[:2]
            new_h, new_w = compute_resize_sizes(h, w, self.frame_size, self.max_aspect_ratio)
            frames = np.stack([cv2.resize(frame, (new_w, new_h)) for frame in frames])
            video = torch.from_numpy(frames).permute(0, 3, 1, 2)
        elif frames.ndim == 5:
            h, w = frames[0, 0].shape[:2]
            new_h, new_w = compute_resize_sizes(h, w, self.frame_size, self.max_aspect_ratio)
            frames = np.stack([
                np.stack([cv2.resize(view, (new_w, new_h)) for view in timestep], axis=0)
                for timestep in frames
            ], axis=0)
            video = torch.from_numpy(frames).permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"expected frames [T,H,W,3] or [T,V,H,W,3], got {frames.shape}")
        return (
            video,
            sampled_timestamps - sampled_timestamps[0],
            instruction,
            None,
            idx,
        )

