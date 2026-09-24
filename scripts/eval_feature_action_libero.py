#!/usr/bin/env python3
"""Closed-loop LIBERO evaluator for DeltaWorld-conditioned ActionDiT."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
import importlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from feature_conditioned_action import (
    ActionKVCache, ConditionBundle,
    DeltaWorldFeatureActionModel, DeltaWorldTokenRollout, T5InstructionEncoder,
)
from feature_conditioned_action.action_kv import LayerKV
from feature_conditioned_action.libero_plus_subset import (
    LIBERO_PLUS_CATEGORIES,
    select_balanced_libero_plus_tasks,
    select_official_libero_plus_tasks,
    summarize_classified_results,
)
from starwam.action_model import load_action_dit_init
from starwam.modules.action_dit import ActionDiT
from starwam.modules.scheduler import FlowMatchScheduler
LOG = logging.getLogger("feature_action_libero")
CAMERA_KEYS = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
}
DUMMY_ACTION = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
# LIBERO / LeRobot default episode lengths. LIBERO-Long is registered as
# ``libero_10`` and uses 520 steps; passing --max-steps explicitly still
# overrides these suite defaults.
LIBERO_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_90": 400,
    "libero_10": 520,
}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON without ever exposing a truncated destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def update_partial_results(
    results: dict[str, Any], output: Path, *, status: str = "running"
) -> None:
    """Persist completed episodes so a native simulator crash is resumable."""
    episodes = [
        episode
        for task in results.get("tasks", {}).values()
        for episode in task.get("episodes", [])
    ]
    successes = sum(int(bool(episode.get("success"))) for episode in episodes)
    trials = len(episodes)
    results.update(
        success_rate=successes / max(trials, 1),
        successes=successes,
        trials=trials,
        libero_plus_summary=summarize_classified_results(results.get("tasks", {})),
        progress={
            "status": status,
            "completed_trials": trials,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    atomic_write_json(output / "results.partial.json", results)


def restore_partial_results(
    path: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    """Load a partial run, rejecting resume with different evaluation settings."""
    restored = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"saved": restored.get(key), "requested": value}
        for key, value in expected.items()
        if restored.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"cannot resume {path}: evaluation settings changed: {mismatches}"
        )
    if not isinstance(restored.get("tasks"), dict):
        raise ValueError(f"cannot resume {path}: missing tasks dictionary")
    return restored


@dataclass
class PendingWorldProbe:
    """A prediction waiting for its aligned closed-loop observations."""

    json_path: Path
    record: dict[str, Any]
    future_tokens: torch.Tensor
    predicted_vfm: torch.Tensor | None
    current_vfm_features: torch.Tensor
    expected_steps: tuple[int, ...]
    observed_frames: dict[int, np.ndarray]
    evaluated_prefix_horizons: set[int] = field(default_factory=set)


def drop_future_tokens_from_condition(condition: ConditionBundle) -> ConditionBundle:
    """Ablation-only path: hide DeltaWorld future tokens from ActionDiT.

    The current-frame spatial anchor, native text/proprio tokens, and optional
    executed-action history stay unchanged.  This measures whether ActionDiT is
    actually using predicted future DeltaWorld tokens without requiring a
    separate checkpoint whose ``world_prediction_steps`` is zero.
    """
    cache = condition.kv_cache
    future = int(cache.num_future_tokens)
    if future <= 0:
        return condition
    anchor = int(cache.num_anchor_tokens)
    start = anchor
    end = anchor + future

    def _drop_sequence(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor[..., :start, :], tensor[..., end:, :]], dim=-2)

    layers = tuple(
        LayerKV(
            key=_drop_sequence(layer.key),
            value=_drop_sequence(layer.value),
        )
        for layer in cache.layers
    )
    key_mask = None
    if cache.key_mask is not None:
        key_mask = torch.cat([cache.key_mask[:, :start], cache.key_mask[:, end:]], dim=1)
    attention_bias = None
    if cache.attention_bias is not None:
        attention_bias = torch.cat(
            [cache.attention_bias[:, :start], cache.attention_bias[:, end:]],
            dim=1,
        )
    new_cache = ActionKVCache(
        layers=layers,
        key_mask=key_mask,
        num_anchor_tokens=cache.num_anchor_tokens,
        num_future_tokens=0,
        num_native_tokens=cache.num_native_tokens,
        num_action_history_tokens=cache.num_action_history_tokens,
        attention_bias=attention_bias,
        history_layers=cache.history_layers,
        history_key_mask=cache.history_key_mask,
        condition_hidden=(
            torch.cat(
                [
                    cache.condition_hidden[:, :start],
                    cache.condition_hidden[:, end:],
                ],
                dim=1,
            )
            if cache.condition_hidden is not None else None
        ),
    )
    empty_future = condition.future_tokens[:, :0]
    empty_mask = condition.future_mask[:, :0]
    return replace(
        condition,
        future_tokens=empty_future,
        future_mask=empty_mask,
        kv_cache=new_cache,
    )


def instantiate_config(value: Any) -> Any:
    """Instantiate a class_path/init_args subtree without importing training wrappers."""
    if isinstance(value, list):
        return [instantiate_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "class_path" not in value:
        return {key: instantiate_config(item) for key, item in value.items()}
    module, _, name = value["class_path"].rpartition(".")
    cls = getattr(importlib.import_module(module), name)
    kwargs = {
        key: instantiate_config(item)
        for key, item in value.get("init_args", {}).items()
    }
    return cls(**kwargs)


def dataset_args(config: dict[str, Any]) -> dict[str, Any]:
    return config["data"]["init_args"]["train_dataset_cfg"]["init_args"]


def load_matching_state_dict(module: nn.Module, checkpoint: dict[str, Any]) -> None:
    """Load same-name, same-shape tensors from a checkpoint.

    This mirrors the small subset of ``training.base.load_sd`` needed for
    evaluation, but intentionally does not import Lightning.  Frozen DeltaWorld
    is restored from its own checkpoint, while the action checkpoint later loads
    only the trainable action/adaptor/task/proprio keys.
    """
    state_dict = checkpoint.get("state_dict", checkpoint)
    target = getattr(module, "_orig_mod", module)
    current = target.state_dict()
    used, mismatched = {}, []
    for key, tensor in state_dict.items():
        if key not in current:
            continue
        if tensor.shape != current[key].shape:
            mismatched.append(key)
            continue
        used[key] = tensor
    missing_trainable = [
        name for name, parameter in target.named_parameters(remove_duplicate=False)
        if parameter.requires_grad and name not in used
    ]
    allowed_missing = {"camera_embedding.weight"}
    allowed_prefixes = (
        "predictor.film_generator.",
        "task_film_encoder.projection.",
    )

    def _is_allowed_missing(name: str) -> bool:
        return name in allowed_missing or name.startswith(allowed_prefixes)

    critical_missing = [name for name in missing_trainable if not _is_allowed_missing(name)]
    tolerated_missing = [name for name in missing_trainable if _is_allowed_missing(name)]
    if tolerated_missing:
        LOG.warning(
            "DeltaWorld checkpoint is missing newly initialized trainable parameters %s; "
            "keeping their identity/random initialization for evaluation.",
            tolerated_missing[:20],
        )
    if critical_missing or mismatched:
        raise RuntimeError(
            "failed to restore DeltaWorld checkpoint: "
            f"missing_trainable={critical_missing[:20]}, mismatched={mismatched[:20]}"
        )
    target.load_state_dict(used, strict=False)


class EvalFeatureConditionedActionPolicy(nn.Module):
    """Inference-only wrapper matching the training checkpoint key layout.

    The training class is a LightningModule.  This eval wrapper deliberately
    keeps the same trainable attribute names (``network``, ``instruction_encoder``,
    ``proprio_encoder``) so action checkpoints load without requiring
    Lightning in simulator / deployment environments.
    """

    def __init__(
        self,
        delta_world: nn.Module,
        world_ckpt_path: str,
        action_dim: int = 7,
        action_horizon: int = 16,
        action_hidden_dim: int = 768,
        action_num_layers: int = 6,
        action_num_heads: int = 12,
        action_head_dim: int = 64,
        action_ffn_ratio: int = 4,
        action_text_dim: int = 256,
        action_freq_dim: int = 256,
        action_eps: float = 1e-6,
        action_init_path: str | None = None,
        action_head_init: str = "random",
        num_world_samples: int = 1,
        freeze_world: bool = True,
        context_frames: int = 4,
        world_prediction_steps: int = 3,
        world_time_delta: float = 0.1,
        world_condition_mode: str = "delta_tokens",
        text_encoder_name: str = "google-t5/t5-base",
        text_encoder_max_length: int = 64,
        freeze_text_encoder: bool = True,
        proprio_dim: int = 9,
        action_history_steps: int = 0,
        action_history_dropout: float = 0.2,
        action_history_attention_mode: str = "separate_gated",
        action_history_gate_init: float = 0.01,
        scheduler_num_train_timesteps: int = 1000,
        scheduler_train_shift: float = 5.0,
        scheduler_infer_shift: float = 5.0,
        **unused_training_args: Any,
    ) -> None:
        super().__init__()
        del scheduler_infer_shift, unused_training_args
        if action_hidden_dim != action_num_heads * action_head_dim:
            raise ValueError(
                "action_hidden_dim must equal action_num_heads * action_head_dim, "
                f"got {action_hidden_dim} vs {action_num_heads}*{action_head_dim}"
            )
        if proprio_dim < 0:
            raise ValueError("proprio_dim must be non-negative")

        world_path = Path(world_ckpt_path)
        if not world_path.is_file():
            raise FileNotFoundError(f"DeltaWorld checkpoint not found: {world_path}")
        world_checkpoint = torch.load(
            world_path, map_location="cpu", weights_only=False, mmap=True
        )
        load_matching_state_dict(delta_world, world_checkpoint)

        action_dit = ActionDiT(
            hidden_dim=action_hidden_dim,
            action_dim=action_dim,
            ffn_dim=action_hidden_dim * action_ffn_ratio,
            text_dim=action_text_dim,
            freq_dim=action_freq_dim,
            eps=action_eps,
            num_heads=action_num_heads,
            attn_head_dim=action_head_dim,
            num_layers=action_num_layers,
            max_seq_len=max(256, action_horizon * 2),
            use_gradient_checkpointing=False,
        )
        load_action_dit_init(action_dit, action_init_path, head_init=action_head_init)
        world_rollout = DeltaWorldTokenRollout(
            delta_world,
            num_samples=num_world_samples,
            freeze_world=freeze_world,
        )
        self.network = DeltaWorldFeatureActionModel(
            world_rollout,
            action_dit,
            action_scheduler=FlowMatchScheduler(
                num_train_timesteps=scheduler_num_train_timesteps,
                shift=scheduler_train_shift,
            ),
            context_frames=context_frames,
            world_prediction_steps=world_prediction_steps,
            world_time_delta=world_time_delta,
            world_condition_mode=world_condition_mode,
            include_text=True,
            action_history_steps=action_history_steps,
            action_history_dropout=action_history_dropout,
            action_history_attention_mode=action_history_attention_mode,
            action_history_gate_init=action_history_gate_init,
        )
        self.action_horizon = int(action_horizon)
        self.proprio_dim = int(proprio_dim)
        self.instruction_encoder = T5InstructionEncoder(
            model_name_or_path=text_encoder_name,
            output_dim=action_text_dim,
            max_length=text_encoder_max_length,
            freeze_encoder=freeze_text_encoder,
        )
        self.proprio_encoder = (
            nn.Sequential(
                nn.LayerNorm(proprio_dim),
                nn.Linear(proprio_dim, action_text_dim),
                nn.SiLU(),
                nn.Linear(action_text_dim, action_text_dim),
            )
            if proprio_dim > 0 else None
        )

    def _conditioning_tokens(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        context, mask = self.instruction_encoder(batch["instruction"])
        if self.proprio_encoder is not None:
            proprio = batch["proprio"]
            if proprio.shape[-1] != self.proprio_dim:
                raise ValueError(
                    f"expected proprio_dim={self.proprio_dim}, got {proprio.shape[-1]}"
                )
            proprio_token = self.proprio_encoder(proprio).unsqueeze(1)
            if proprio_token.shape[0] != context.shape[0]:
                raise ValueError("instruction/proprio batch size mismatch")
            context = torch.cat([context, proprio_token], dim=1)
            proprio_mask = torch.ones(
                proprio_token.shape[:2], dtype=torch.bool, device=context.device
            )
            mask = torch.cat([mask, proprio_mask], dim=1)
        return context, mask


class LiberoObservationAdapter:
    """Apply the camera selection/layout and preprocessing stored in training YAML."""

    def __init__(self, cameras, camera_layout: str, frame_size) -> None:
        self.cameras = tuple(cameras)
        self.layout = camera_layout
        self.size = ((frame_size, frame_size) if isinstance(frame_size, int) else frame_size)
        if not self.cameras:
            raise ValueError("at least one camera is required")
        if self.layout not in {"first", "horizontal", "vertical", "multi_view"}:
            raise ValueError(f"unsupported camera_layout={self.layout!r}")
        unknown = set(self.cameras) - set(CAMERA_KEYS)
        if unknown:
            raise ValueError(f"no simulator mapping for cameras {sorted(unknown)}")

    def frame(self, observation: dict[str, Any]) -> np.ndarray:
        images = []
        for camera in self.cameras:
            key = CAMERA_KEYS[camera]
            if key not in observation:
                raise KeyError(f"simulator observation is missing {key!r}")
            # LIBERO HDF5 demos and OffScreenRenderEnv observations use the
            # same raw OpenGL tensor convention.  Do not flip policy inputs here:
            # scripts/libero_sanity_checks.py verifies raw simulator tensors match
            # HDF5 tensors with low MAE.  For human-readable videos, flip in
            # display_frame() only; in this workspace flip_both is usually upright.
            images.append(np.ascontiguousarray(observation[key]))
        height, width = map(int, self.size)
        if self.layout == "multi_view":
            views = [cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
                     for image in images]
            image = np.stack(views, axis=0)
        elif self.layout == "first" or len(images) == 1:
            image = cv2.resize(images[0], (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            image = np.concatenate(images, axis=1 if self.layout == "horizontal" else 0)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.asarray(np.clip(image, 0, 255), dtype=np.uint8)

    @staticmethod
    def proprio(observation: dict[str, Any]) -> np.ndarray:
        # Exact HDF5 robot_states order used during training.
        state = np.concatenate([
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            np.asarray(observation["robot0_eef_quat"], dtype=np.float32),
        ])
        if state.shape != (9,):
            raise ValueError(f"expected 9-D robot state, got {state.shape}")
        return state


def _unwrap_libero_env(env: Any) -> Any:
    """Follow common wrappers until the LIBERO robosuite env is reached."""
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        next_env = getattr(current, "env")
        if next_env is current:
            break
        current = next_env
    return current


def _object_state_dict(env: Any) -> dict[str, Any]:
    raw_env = _unwrap_libero_env(env)
    states = getattr(raw_env, "object_states_dict", None)
    return states if isinstance(states, dict) else {}


def _object_position(state: Any) -> np.ndarray | None:
    try:
        geom_state = state.get_geom_state()
    except Exception:
        return None
    if not isinstance(geom_state, dict) or "pos" not in geom_state:
        return None
    pos = np.asarray(geom_state["pos"], dtype=np.float64).reshape(-1)
    return pos[:3] if pos.size >= 3 else None


def _libero_object_positions(env: Any) -> dict[str, list[float]]:
    positions: dict[str, list[float]] = {}
    for name, state in _object_state_dict(env).items():
        pos = _object_position(state)
        if pos is not None:
            positions[name] = pos.tolist()
    return positions


def _goal_on_pair(env: Any, bowl_names: list[str], plate_names: list[str]) -> tuple[str | None, str | None]:
    raw_env = _unwrap_libero_env(env)
    parsed = getattr(raw_env, "parsed_problem", None)
    goal_state = parsed.get("goal_state", []) if isinstance(parsed, dict) else []
    for predicate in goal_state:
        if not predicate:
            continue
        if str(predicate[0]).lower() != "on" or len(predicate) < 3:
            continue
        subject, target = str(predicate[1]), str(predicate[2])
        if subject in bowl_names and target in plate_names:
            return subject, target
    return (bowl_names[0] if bowl_names else None, plate_names[0] if plate_names else None)


def _check_grasped(raw_env: Any, object_name: str) -> bool:
    try:
        robot = raw_env.robots[0]
        return bool(raw_env._check_grasp(
            gripper=robot.gripper,
            object_geoms=raw_env.get_object(object_name),
        ))
    except Exception:
        return False


def collect_success_diagnostics(
    env: Any,
    observation: dict[str, Any],
    *,
    step: int,
    action: np.ndarray,
    done: bool,
    initial_positions: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Collect task-object geometry diagnostics without changing LIBERO success."""
    raw_env = _unwrap_libero_env(env)
    states = _object_state_dict(raw_env)
    bowl_names = sorted(name for name in states if "bowl" in name.lower())
    plate_names = sorted(name for name in states if "plate" in name.lower())
    goal_bowl, goal_plate = _goal_on_pair(raw_env, bowl_names, plate_names)
    plate_state = states.get(goal_plate) if goal_plate is not None else None
    plate_pos = _object_position(plate_state) if plate_state is not None else None
    eef_pos = np.asarray(observation.get("robot0_eef_pos", []), dtype=np.float64).reshape(-1)
    if eef_pos.size < 3:
        eef_pos = None
    else:
        eef_pos = eef_pos[:3]
    gripper_qpos = np.asarray(observation.get("robot0_gripper_qpos", []), dtype=np.float64).reshape(-1)
    gripper_opening = float(np.abs(gripper_qpos).sum()) if gripper_qpos.size else None
    initial_positions = initial_positions or {}

    bowl_records: dict[str, dict[str, Any]] = {}
    for name in bowl_names:
        state = states.get(name)
        pos = _object_position(state)
        initial = np.asarray(initial_positions.get(name, []), dtype=np.float64).reshape(-1)
        displacement = (
            float(np.linalg.norm(pos - initial[:3]))
            if pos is not None and initial.size >= 3 else None
        )
        eef_distance = (
            float(np.linalg.norm(pos - eef_pos))
            if pos is not None and eef_pos is not None else None
        )
        xy_distance = (
            float(np.linalg.norm(pos[:2] - plate_pos[:2]))
            if pos is not None and plate_pos is not None else None
        )
        z_delta = (
            float(pos[2] - plate_pos[2])
            if pos is not None and plate_pos is not None else None
        )
        contact = False
        official_on = False
        if state is not None and plate_state is not None:
            try:
                contact = bool(plate_state.check_contact(state))
            except Exception:
                contact = False
            try:
                official_on = bool(plate_state.check_ontop(state))
            except Exception:
                official_on = False
        bowl_records[name] = {
            "position": pos.tolist() if pos is not None else None,
            "is_goal_bowl": name == goal_bowl,
            "grasped": _check_grasped(raw_env, name),
            "displacement_from_reset": displacement,
            "eef_distance": eef_distance,
            "plate_contact": contact,
            "plate_xy_distance": xy_distance,
            "plate_z_delta": z_delta,
            "plate_z_ok": bool(z_delta is not None and z_delta >= 0.0),
            "official_on_plate": official_on,
        }

    grasped = [name for name, record in bowl_records.items() if record["grasped"]]
    if len(grasped) == 1:
        active_bowl, active_reason = grasped[0], "grasped"
    elif len(grasped) > 1:
        active_bowl = min(grasped, key=lambda name: bowl_records[name]["eef_distance"] or float("inf"))
        active_reason = "nearest_grasped"
    else:
        movable = [
            (record["displacement_from_reset"] or 0.0, name)
            for name, record in bowl_records.items()
        ]
        best_displacement, best_name = max(movable, default=(0.0, None))
        if best_name is not None and best_displacement > 0.01:
            active_bowl, active_reason = best_name, "most_displaced"
        else:
            active_bowl = min(
                bowl_records,
                key=lambda name: bowl_records[name]["eef_distance"] or float("inf"),
                default=None,
            )
            active_reason = "nearest_eef" if active_bowl is not None else None

    active_record = bowl_records.get(active_bowl) if active_bowl is not None else None
    return {
        "step": int(step),
        "done": bool(done),
        "action": np.asarray(action, dtype=np.float32).reshape(-1).tolist(),
        "goal_bowl": goal_bowl,
        "goal_plate": goal_plate,
        "active_bowl": active_bowl,
        "active_bowl_reason": active_reason,
        "active_is_goal_bowl": bool(active_bowl == goal_bowl) if active_bowl is not None else None,
        "active_plate_contact": active_record["plate_contact"] if active_record else None,
        "active_plate_xy_distance": active_record["plate_xy_distance"] if active_record else None,
        "active_plate_z_delta": active_record["plate_z_delta"] if active_record else None,
        "active_plate_z_ok": active_record["plate_z_ok"] if active_record else None,
        "active_official_on_plate": active_record["official_on_plate"] if active_record else None,
        "goal_plate_contact": bowl_records.get(goal_bowl, {}).get("plate_contact"),
        "goal_plate_xy_distance": bowl_records.get(goal_bowl, {}).get("plate_xy_distance"),
        "goal_plate_z_delta": bowl_records.get(goal_bowl, {}).get("plate_z_delta"),
        "goal_official_on_plate": bowl_records.get(goal_bowl, {}).get("official_on_plate"),
        "gripper_opening": gripper_opening,
        "bowls": bowl_records,
    }


def summarize_success_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_records": 0}
    active_counts: dict[str, int] = {}
    for record in records:
        name = record.get("active_bowl")
        if name:
            active_counts[name] = active_counts.get(name, 0) + 1
    goal_xy = [
        record["goal_plate_xy_distance"] for record in records
        if record.get("goal_plate_xy_distance") is not None
    ]
    wrong_on_steps = 0
    for record in records:
        goal_bowl = record.get("goal_bowl")
        for name, bowl in record.get("bowls", {}).items():
            if name != goal_bowl and bowl.get("official_on_plate"):
                wrong_on_steps += 1
                break
    return {
        "num_records": len(records),
        "final": records[-1],
        "active_bowl_step_counts": active_counts,
        "active_goal_bowl_steps": sum(bool(record.get("active_is_goal_bowl")) for record in records),
        "goal_plate_contact_steps": sum(bool(record.get("goal_plate_contact")) for record in records),
        "goal_official_on_plate_steps": sum(bool(record.get("goal_official_on_plate")) for record in records),
        "wrong_bowl_official_on_plate_steps": wrong_on_steps,
        "min_goal_plate_xy_distance": min(goal_xy) if goal_xy else None,
        "last_goal_plate_xy_distance": goal_xy[-1] if goal_xy else None,
    }


def oracle_replan_decision(
    env: Any,
    observation: dict[str, Any],
    *,
    far_replan_steps: int,
    place_replan_steps: int,
    place_xy_threshold: float,
    require_active_goal: bool = True,
    require_grasped: bool = True,
) -> dict[str, Any]:
    """Privileged simulator oracle for deciding how long to execute a chunk.

    This is an evaluation ablation only.  It uses simulator object geometry to
    detect whether the rollout has entered the placement phase.
    """
    diagnostics = collect_success_diagnostics(
        env,
        observation,
        step=-1,
        action=np.zeros_like(DUMMY_ACTION),
        done=False,
        initial_positions=None,
    )
    goal_bowl = diagnostics.get("goal_bowl")
    goal = diagnostics.get("bowls", {}).get(goal_bowl, {}) if goal_bowl else {}
    xy = diagnostics.get("goal_plate_xy_distance")
    active_is_goal = bool(diagnostics.get("active_is_goal_bowl"))
    grasped = bool(goal.get("grasped"))
    near_plate = xy is not None and xy < place_xy_threshold
    place_phase = bool(near_plate)
    if require_active_goal:
        place_phase = place_phase and active_is_goal
    if require_grasped:
        place_phase = place_phase and grasped
    selected = int(place_replan_steps if place_phase else far_replan_steps)
    return {
        "selected_replan_steps": selected,
        "place_phase": place_phase,
        "near_plate": near_plate,
        "place_xy_threshold": float(place_xy_threshold),
        "goal_plate_xy_distance": xy,
        "active_bowl": diagnostics.get("active_bowl"),
        "active_bowl_reason": diagnostics.get("active_bowl_reason"),
        "active_is_goal_bowl": active_is_goal,
        "goal_bowl_grasped": grasped,
        "goal_plate_contact": diagnostics.get("goal_plate_contact"),
    }


def task9_world_consistency_phase(
    env: Any,
    observation: dict[str, Any],
    *,
    far_distance: float,
    disabled_after_grasp: bool,
) -> tuple[dict[str, Any], bool]:
    """Privileged phase gate for the task-9 online-consistency pilot.

    The token-consistency score itself uses only images available online.  This
    helper deliberately uses LIBERO object state for a controlled task-9
    ablation: intervention is allowed only while the end effector is still far
    from the goal bowl, and is permanently disabled once that bowl is grasped.
    A deployable replacement should obtain these two facts from perception and
    gripper/contact state rather than simulator object geometry.
    """
    diagnostics = collect_success_diagnostics(
        env,
        observation,
        step=-1,
        action=np.zeros_like(DUMMY_ACTION),
        done=False,
        initial_positions=None,
    )
    goal_bowl = diagnostics.get("goal_bowl")
    goal_record = diagnostics.get("bowls", {}).get(goal_bowl, {}) if goal_bowl else {}
    grasped = bool(goal_record.get("grasped"))
    eef_distance = goal_record.get("eef_distance")
    latched = bool(disabled_after_grasp or grasped)
    if latched:
        active = False
        reason = "disabled_after_goal_bowl_grasp"
    elif eef_distance is None:
        active = False
        reason = "goal_bowl_distance_unavailable"
    elif float(eef_distance) <= float(far_distance):
        active = False
        reason = "goal_bowl_already_near"
    else:
        active = True
        reason = "pregrasp_far_from_goal_bowl"
    return ({
        "phase_active": active,
        "phase_reason": reason,
        "goal_bowl": goal_bowl,
        "goal_bowl_grasped": grasped,
        "goal_bowl_eef_distance": (
            float(eef_distance) if eef_distance is not None else None
        ),
        "far_distance_threshold": float(far_distance),
        "disabled_after_grasp": latched,
    }, latched)


def world_consistency_replan_decision(
    event: dict[str, Any],
    phase: dict[str, Any],
    *,
    policy_step: int,
    soft_threshold: float,
    hard_threshold: float,
    min_progress: float,
    cooldown_steps: int,
    max_triggers: int,
    max_policy_step: int,
    last_trigger_step: int | None,
    trigger_count: int,
    episode_done: bool,
) -> dict[str, Any]:
    """Conservative two-level online-consistency intervention rule."""
    score = event.get("pred_gt_token_cosine")
    prediction_distance = event.get("prediction_goal_bowl_eef_distance")
    current_distance = phase.get("goal_bowl_eef_distance")
    distance_progress = None
    if prediction_distance is not None and current_distance is not None:
        # Positive means the end effector moved closer to the intended bowl.
        distance_progress = float(prediction_distance) - float(current_distance)
    soft_low = bool(score is not None and float(score) < soft_threshold)
    hard_low = bool(score is not None and float(score) < hard_threshold)
    insufficient_progress = bool(
        distance_progress is None or distance_progress < min_progress
    )
    cooldown_remaining = 0
    if last_trigger_step is not None:
        cooldown_remaining = max(
            0, int(cooldown_steps) - (int(policy_step) - int(last_trigger_step))
        )

    triggered = False
    if score is None:
        reason = "metric_unavailable"
    elif episode_done:
        reason = "episode_already_done"
    elif not phase.get("phase_active", False):
        reason = str(phase.get("phase_reason", "phase_inactive"))
    elif policy_step > max_policy_step:
        reason = "outside_early_policy_window"
    elif trigger_count >= max_triggers:
        reason = "trigger_budget_exhausted"
    elif not soft_low:
        reason = "token_consistency_ok"
    elif not hard_low and not insufficient_progress:
        reason = "soft_low_but_goal_distance_improving"
    elif cooldown_remaining > 0:
        reason = "trigger_cooldown"
    else:
        triggered = True
        reason = (
            "severe_token_inconsistency"
            if hard_low else "soft_low_and_insufficient_goal_progress"
        )
    return {
        "triggered": triggered,
        "decision_reason": reason,
        "soft_threshold": float(soft_threshold),
        "hard_threshold": float(hard_threshold),
        "soft_low": soft_low,
        "hard_low": hard_low,
        "prediction_goal_bowl_eef_distance": prediction_distance,
        "goal_bowl_eef_distance": current_distance,
        "goal_bowl_distance_progress": distance_progress,
        "min_required_progress": float(min_progress),
        "insufficient_progress": insufficient_progress,
        "policy_step": int(policy_step),
        "max_policy_step": int(max_policy_step),
        "cooldown_steps": int(cooldown_steps),
        "cooldown_remaining": int(cooldown_remaining),
        "trigger_count_before_decision": int(trigger_count),
        "max_triggers": int(max_triggers),
    }


def build_task_vocabulary(root: Path) -> dict[str, int]:
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10")
    names = sorted({
        path.stem.removesuffix("_demo")
        for suite in suites for path in (root / suite).glob("*.hdf5")
    })
    if not names:
        raise FileNotFoundError(f"no LIBERO HDF5 files below {root}")
    return {name: index for index, name in enumerate(names)}


def resolve_task_id(task: Any, vocabulary: dict[str, int]) -> int:
    bddl = Path(str(getattr(task, "bddl_file", ""))).stem
    candidates = [
        str(getattr(task, "name", "")),
        str(getattr(task, "language", "")).strip().lower().replace(" ", "_"),
        bddl,
        bddl.removesuffix("_demo"),
    ]
    for candidate in candidates:
        if candidate in vocabulary:
            return vocabulary[candidate]
    raise KeyError(f"cannot map task to training task id; tried {candidates}")


def load_policy(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    world_prediction_steps_override: int | None = None,
    world_checkpoint_override: Path | None = None,
):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    init_args = dict(model_cfg.get("init_args", {}))
    if "delta_world" not in init_args:
        raise KeyError("model.init_args.delta_world is required for eval-only loading")
    configured_world_checkpoint = init_args.get("world_ckpt_path")
    if world_checkpoint_override is not None:
        world_checkpoint_override = world_checkpoint_override.expanduser().resolve()
        if not world_checkpoint_override.is_file():
            raise FileNotFoundError(
                f"DeltaWorld checkpoint override not found: {world_checkpoint_override}"
            )
        init_args["world_ckpt_path"] = str(world_checkpoint_override)
        LOG.info(
            "overriding frozen DeltaWorld checkpoint for eval: %s -> %s",
            configured_world_checkpoint, world_checkpoint_override,
        )
    if world_prediction_steps_override is not None:
        if world_prediction_steps_override <= 0:
            raise ValueError("world_prediction_steps_override must be positive")
        old_steps = init_args.get("world_prediction_steps")
        init_args["world_prediction_steps"] = int(world_prediction_steps_override)
        LOG.info(
            "overriding world_prediction_steps for eval: %s -> %s",
            old_steps, init_args["world_prediction_steps"],
        )
    delta_world_cfg = init_args.pop("delta_world")
    delta_world = instantiate_config(delta_world_cfg)
    policy = EvalFeatureConditionedActionPolicy(delta_world=delta_world, **init_args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = policy.load_state_dict(state_dict, strict=False)
    allowed_unexpected = {"network.kv_adapter.text_type_embedding"}
    unexpected = [
        key for key in incompatible.unexpected_keys
        if key not in allowed_unexpected
    ]
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:10]}")
    tolerated_unexpected = [
        key for key in incompatible.unexpected_keys
        if key in allowed_unexpected
    ]
    if tolerated_unexpected:
        LOG.warning(
            "ignoring legacy checkpoint keys no longer used by native text/proprio path: %s",
            tolerated_unexpected,
        )
    trainable_names = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    missing_trainable = [name for name in incompatible.missing_keys if name in trainable_names]
    if missing_trainable:
        raise RuntimeError(f"missing trainable checkpoint keys: {missing_trainable[:20]}")
    action_checkpoint_world = checkpoint.get("external_world_checkpoint")
    policy.eval_world_checkpoint = str(init_args.get("world_ckpt_path"))
    policy.action_checkpoint_world = action_checkpoint_world
    policy.world_checkpoint_overridden = world_checkpoint_override is not None
    policy.to(device).eval()
    if policy.network.world_rollout.training or policy.network.world_rollout.delta_world.training:
        raise RuntimeError("frozen DeltaWorld rollout must remain in eval mode during rollout")
    LOG.info(
        "loaded eval-only policy %s global_step=%s action_checkpoint_world=%s "
        "effective_world=%s overridden=%s",
        checkpoint_path,
        checkpoint.get("global_step"),
        action_checkpoint_world,
        policy.eval_world_checkpoint,
        policy.world_checkpoint_overridden,
    )
    return policy, config


def build_executed_action_history(
    executed_actions, max_history_steps: int, action_dim: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Right-align only commands that have actually passed through env.step()."""
    if max_history_steps < 0:
        raise ValueError("max_history_steps must be non-negative")
    if max_history_steps == 0:
        return None, None
    history = np.zeros((max_history_steps, action_dim), dtype=np.float32)
    mask = np.zeros(max_history_steps, dtype=np.bool_)
    valid = list(executed_actions or [])[-max_history_steps:]
    if valid:
        array = np.asarray(valid, dtype=np.float32)
        if array.shape != (len(valid), action_dim):
            raise ValueError(
                f"executed action history must be [H,{action_dim}], got {array.shape}"
            )
        history[-len(valid):] = array
        mask[-len(valid):] = True
    return torch.from_numpy(history).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)


def _disagreement_gate(value: float, low: float, high: float) -> float:
    """Smoothly map a disagreement magnitude to stale-chunk reliability."""
    if value <= low:
        return 1.0
    if value >= high:
        return 0.0
    unit = (high - value) / (high - low)
    return float(unit * unit * (3.0 - 2.0 * unit))


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return 1.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def summarize_temporal_disagreement(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-control-step disagreement diagnostics."""
    overlap = [record for record in records if record["compared_old_chunks"] > 0]

    def mean_present(key: str) -> float | None:
        values = [float(record[key]) for record in overlap if record.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "control_steps": len(records),
        "overlap_steps": len(overlap),
        "overlap_fraction": len(overlap) / max(len(records), 1),
        "position_disagreement_mean": mean_present("position_disagreement_mean"),
        "rotation_disagreement_mean": mean_present("rotation_disagreement_mean"),
        "direction_cosine_mean": mean_present("direction_cosine_min"),
        "direction_conflict_rate": (
            sum(bool(record["direction_conflict"]) for record in overlap)
            / max(len(overlap), 1)
        ),
        "gripper_conflict_rate": (
            sum(bool(record["gripper_conflict"]) for record in overlap)
            / max(len(overlap), 1)
        ),
        "old_gate_mean": mean_present("old_gate_mean"),
        "new_only_rate": (
            sum(bool(record["new_only"]) for record in overlap)
            / max(len(overlap), 1)
        ),
    }


def temporal_ensemble_action(
    chunks: list[tuple[int, np.ndarray]],
    current_step: int,
    decay: float,
    *,
    disagreement_aware: bool = False,
    disagreement_window: int = 4,
    position_low: float = 0.08,
    position_high: float = 0.20,
    rotation_low: float = 0.10,
    rotation_high: float = 0.30,
    direction_cosine_threshold: float = 0.0,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Fuse overlapping chunk predictions aligned to one absolute control step.

    Each stale chunk is compared with the newest chunk over a short aligned
    future window. Only stale weights are gated down when plans conflict.
    """
    if current_step < 0:
        raise ValueError("current_step must be non-negative")
    if decay < 0:
        raise ValueError("temporal ensemble decay must be non-negative")
    if disagreement_window <= 0:
        raise ValueError("disagreement_window must be positive")
    if not 0 <= position_low < position_high:
        raise ValueError("position thresholds must satisfy 0 <= low < high")
    if not 0 <= rotation_low < rotation_high:
        raise ValueError("rotation thresholds must satisfy 0 <= low < high")
    if not -1.0 <= direction_cosine_threshold <= 1.0:
        raise ValueError("direction cosine threshold must be in [-1,1]")

    active: list[dict[str, Any]] = []
    for start_step, chunk in chunks:
        chunk = np.asarray(chunk, dtype=np.float32)
        offset = current_step - int(start_step)
        if chunk.ndim != 2:
            raise ValueError(f"action chunk must be [H,A], got {chunk.shape}")
        if chunk.shape[1] < 7:
            raise ValueError(
                f"action chunk must have at least 7 action dims, got {chunk.shape}"
            )
        if 0 <= offset < len(chunk):
            active.append({
                "start": int(start_step),
                "chunk": chunk,
                "offset": offset,
                "action": chunk[offset],
                "age": offset,
                "gate": 1.0,
            })
    if not active:
        raise RuntimeError(f"no temporal-ensemble action covers step {current_step}")

    newest = max(active, key=lambda item: item["start"])
    comparisons: list[dict[str, Any]] = []
    if disagreement_aware:
        for item in active:
            if item is newest:
                continue
            length = min(
                disagreement_window,
                len(item["chunk"]) - item["offset"],
                len(newest["chunk"]) - newest["offset"],
            )
            old_seq = item["chunk"][item["offset"] : item["offset"] + length]
            new_seq = newest["chunk"][newest["offset"] : newest["offset"] + length]
            position_disagreement = float(
                np.sqrt(np.mean(np.square(old_seq[:, :3] - new_seq[:, :3])))
            )
            rotation_disagreement = float(
                np.sqrt(np.mean(np.square(old_seq[:, 3:6] - new_seq[:, 3:6])))
            )
            position_cosine = _safe_cosine(old_seq[:, :3], new_seq[:, :3])
            rotation_cosine = _safe_cosine(old_seq[:, 3:6], new_seq[:, 3:6])
            direction_cosine = min(position_cosine, rotation_cosine)
            direction_conflict = direction_cosine < direction_cosine_threshold
            gripper_conflict = bool(
                np.any(np.signbit(old_seq[:, 6]) != np.signbit(new_seq[:, 6]))
            )
            gate = min(
                _disagreement_gate(position_disagreement, position_low, position_high),
                _disagreement_gate(rotation_disagreement, rotation_low, rotation_high),
            )
            if direction_conflict or gripper_conflict:
                gate = 0.0
            item["gate"] = float(gate)
            comparisons.append({
                "position_disagreement": position_disagreement,
                "rotation_disagreement": rotation_disagreement,
                "direction_cosine": direction_cosine,
                "direction_conflict": direction_conflict,
                "gripper_conflict": gripper_conflict,
                "gate": float(gate),
            })

    actions = np.stack([item["action"] for item in active], axis=0)
    ages = np.asarray([item["age"] for item in active], dtype=np.float32)
    gates = np.asarray([item["gate"] for item in active], dtype=np.float32)
    weights = np.exp(-float(decay) * ages) * gates
    weights /= weights.sum()
    action = np.sum(actions * weights[:, None], axis=0, dtype=np.float32)
    action[:6] = np.clip(action[:6], -1.0, 1.0)
    action[6] = 1.0 if action[6] >= 0.0 else -1.0
    action = action.astype(np.float32)
    diagnostics = {
        "active_candidates": len(active),
        "compared_old_chunks": len(comparisons),
        "position_disagreement_mean": (
            float(np.mean([item["position_disagreement"] for item in comparisons]))
            if comparisons else None
        ),
        "rotation_disagreement_mean": (
            float(np.mean([item["rotation_disagreement"] for item in comparisons]))
            if comparisons else None
        ),
        "direction_cosine_min": (
            float(min(item["direction_cosine"] for item in comparisons))
            if comparisons else None
        ),
        "direction_conflict": any(item["direction_conflict"] for item in comparisons),
        "gripper_conflict": any(item["gripper_conflict"] for item in comparisons),
        "old_gate_mean": (
            float(np.mean([item["gate"] for item in comparisons]))
            if comparisons else None
        ),
        "new_only": bool(
            comparisons and all(item["gate"] == 0.0 for item in comparisons)
        ),
    }
    return (action, diagnostics) if return_diagnostics else action


def select_consensus_action_candidate(
    chunks: list[np.ndarray],
    *,
    lookahead_steps: int,
) -> tuple[int, dict[str, Any]]:
    """Select the medoid of stochastic retry plans without privileged state.

    Distances use the first six continuous action dimensions. The gripper is
    excluded because a binary sign flip would otherwise dominate the short
    reach-phase comparison.
    """
    if not chunks:
        raise ValueError("at least one retry action chunk is required")
    horizon = min(int(lookahead_steps), *(len(chunk) for chunk in chunks))
    if horizon <= 0:
        raise ValueError("retry candidate lookahead must be positive")
    plans = np.stack([
        np.asarray(chunk[:horizon, :6], dtype=np.float32) for chunk in chunks
    ])
    pairwise = np.mean(
        np.square(plans[:, None] - plans[None, :]), axis=(2, 3)
    )
    # Ignore the zero self-distance so K has no artificial scale dependence.
    if len(chunks) == 1:
        mean_disagreement = np.zeros(1, dtype=np.float32)
    else:
        mean_disagreement = pairwise.sum(axis=1) / (len(chunks) - 1)
    selected = int(np.argmin(mean_disagreement))
    return selected, {
        "num_candidates": len(chunks),
        "lookahead_steps": horizon,
        "selected_candidate": selected,
        "mean_action_disagreement_by_candidate": mean_disagreement.tolist(),
        "selected_mean_action_disagreement": float(mean_disagreement[selected]),
    }


def cache_instruction_conditions(policy, instruction: str):
    """Encode episode-static language conditions once and reuse at replans."""
    device = next(policy.parameters()).device
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        text_context, text_mask = policy.instruction_encoder([instruction])
        world_condition = None
        world_encoder = getattr(
            policy.network.world_rollout.delta_world, "task_film_encoder", None
        )
        if world_encoder is not None:
            world_condition, _ = world_encoder([instruction])
    return text_context, text_mask, world_condition


def predict_action_chunk(
    policy,
    history,
    instruction: str,
    proprio: np.ndarray,
    context_frames: int,
    context_stride: int,
    action_horizon: int,
    inference_steps: int,
    fps: float,
    seed: int | None,
    world_frame_stride: int | None = None,
    *,
    world_probe_dir: Path | None = None,
    world_probe_meta: dict[str, Any] | None = None,
    world_probe_pending: list[PendingWorldProbe] | None = None,
    world_probe_save_visualization: bool = True,
    video_flip: str = "both",
    force_context_tokens: int | None = None,
    executed_actions=None,
    disable_action_history: bool = False,
    disable_world_future_tokens: bool = False,
    cached_text_context: torch.Tensor | None = None,
    cached_text_mask: torch.Tensor | None = None,
    cached_world_condition: torch.Tensor | None = None,
    paper_recorder=None,
) -> np.ndarray:
    if context_stride <= 0:
        raise ValueError(f"context_stride must be positive, got {context_stride}")
    frames = list(history)
    if not frames:
        raise ValueError("need at least one observed frame for warm-start prediction")
    # Warm-start: before the replay buffer contains enough frames for the full
    # [t-6,t-4,t-2,t] grid, attend only to actually observed context tokens.
    # Additionally, the first policy replan after dummy wait actions can force
    # a single context token: wait-step frames are near-duplicates of the reset
    # image and should not masquerade as motion history.
    available_context = min(context_frames, 1 + (len(frames) - 1) // context_stride)
    if force_context_tokens is not None:
        if force_context_tokens <= 0:
            raise ValueError(f"force_context_tokens must be positive, got {force_context_tokens}")
        available_context = min(available_context, int(force_context_tokens))
    # Once enough non-forced frames are available, this exactly recovers the
    # standard fixed 4-context path.
    indices = [
        len(frames) - 1 - (available_context - 1 - i) * context_stride
        for i in range(available_context)
    ]
    stacked = np.stack([frames[i] for i in indices])
    if stacked.ndim == 4:
        video = torch.from_numpy(stacked).permute(0, 3, 1, 2)
    elif stacked.ndim == 5:
        video = torch.from_numpy(stacked).permute(0, 1, 4, 2, 3)
    else:
        raise ValueError(f"expected stacked history [T,H,W,3] or [T,V,H,W,3], got {stacked.shape}")
    device = next(policy.parameters()).device
    proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        if cached_text_context is None or cached_text_mask is None:
            batch = {
                "instruction": [instruction],
                "proprio": proprio_tensor,
            }
            context, mask = policy._conditioning_tokens(batch)
        else:
            context = cached_text_context.to(device)
            mask = cached_text_mask.to(device=device, dtype=torch.bool)
            if policy.proprio_encoder is not None:
                proprio_token = policy.proprio_encoder(proprio_tensor).unsqueeze(1)
                context = torch.cat([context, proprio_token], dim=1)
                proprio_mask = torch.ones(
                    proprio_token.shape[:2], dtype=torch.bool, device=device
                )
                mask = torch.cat([mask, proprio_mask], dim=1)
    action_history, action_history_mask = build_executed_action_history(
        executed_actions,
        policy.network.action_history_steps,
        policy.network.action_dit.action_dim,
    )
    if action_history is not None:
        action_history = action_history.to(device)
        action_history_mask = action_history_mask.to(device)
        if disable_action_history:
            # Ablation-only path: preserve the trained model architecture and
            # checkpoint shape, but make the executed-action memory completely
            # invisible at eval time.  Zeroing values is a defensive no-op in
            # case a future attention path accidentally ignores the mask.
            action_history = torch.zeros_like(action_history)
            action_history_mask = torch.zeros_like(action_history_mask)
    times = torch.tensor(indices, dtype=torch.float32, device=device)[None] / fps
    future_steps = int(policy.network.world_prediction_steps)
    world_stride = int(world_frame_stride if world_frame_stride is not None else context_stride)
    if world_stride <= 0:
        raise ValueError(f"world_frame_stride must be positive, got {world_stride}")
    future_offsets = torch.arange(1, future_steps + 1, dtype=torch.float32, device=device)[None]
    future_times = times[:, -1:] + future_offsets * (world_stride / fps)
    observed = video[None].to(device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=observed.device).manual_seed(seed)

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                enabled=device.type == "cuda"):
        from contextlib import nullcontext
        from scripts.paper_visualization_recording import capture_predictor_attention
        if paper_recorder is not None:
            paper_recorder.start({**(world_probe_meta or {}), "seed": seed,
                                  "instruction": instruction, "world_frame_stride": world_stride})
        capture = (capture_predictor_attention(paper_recorder.predictor)
                   if paper_recorder is not None else nullcontext())
        with capture:
            condition = policy.network.encode_condition(
                observed,
                observed_timestamps=times,
                future_timestamps=future_times,
                text_context=context,
                text_mask=mask,
                generator=generator,
                instruction=(
                    cached_world_condition
                    if cached_world_condition is not None else [instruction]
                ),
                action_history=action_history,
                action_history_mask=action_history_mask,
            )
        if disable_world_future_tokens:
            condition = drop_future_tokens_from_condition(condition)
        if paper_recorder is not None:
            paper_recorder.condition(observed, times, future_times, condition)
        if world_probe_dir is not None:
            meta = dict(world_probe_meta or {})
            pending_probe = save_world_probe(
                world_probe_dir,
                policy,
                observed,
                condition,
                task_id=int(meta.get("task_id", -1)),
                trial=int(meta.get("trial", -1)),
                step=int(meta.get("step", -1)),
                instruction=instruction,
                video_flip=video_flip,
                future_step_stride=world_stride,
                save_visualization=world_probe_save_visualization,
                metadata=meta,
            )
            if world_probe_pending is not None:
                world_probe_pending.append(pending_probe)
        actions = torch.randn(
            observed.shape[0],
            action_horizon,
            policy.network.action_dit.action_dim,
            device=device,
            dtype=policy.network.action_dit.action_encoder.weight.dtype,
            generator=generator,
        )
        timesteps, deltas = policy.network.action_scheduler.build_inference_schedule(
            inference_steps, actions.device, actions.dtype
        )
        for timestep, delta in zip(timesteps, deltas):
            velocity = (
                paper_recorder.velocity(policy.network, actions, timestep.expand(actions.shape[0]), condition)
                if paper_recorder is not None else
                policy.network(actions, timestep.expand(actions.shape[0]), condition)
            )
            actions = policy.network.action_scheduler.step(velocity, delta, actions)
    actions = actions[0].detach().float().cpu().numpy()
    actions[:, :6] = np.clip(actions[:, :6], -1, 1)
    actions[:, 6] = np.where(actions[:, 6] >= 0, 1, -1)
    actions = actions.astype(np.float32)
    if paper_recorder is not None:
        paper_recorder.finish_plan(actions)
    return actions


def configure_libero(home: Path | None) -> None:
    if not home:
        return
    home = home.expanduser().resolve()
    # LIBERO and LIBERO-Plus repos have the layout:
    #   <repo>/libero/libero/__init__.py
    # The import used below is ``from libero.libero import ...``, so sys.path
    # must contain <repo>, not <repo>/libero.  Adding the inner directory first
    # collapses the package one level and makes ``libero.libero`` unimportable.
    path_candidates = (home,) if (home / "libero" / "libero").is_dir() else (home, home / "libero")
    for candidate in reversed(path_candidates):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    config_dir = Path(os.environ.get(
        "LIBERO_CONFIG_PATH", Path.home() / ".cache/deltatok/libero_config"
    ))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    benchmark_root = home / "libero" / "libero"
    config_file = config_dir / "config.yaml"
    if benchmark_root.is_dir() and not config_file.exists():
        config_file.write_text(yaml.safe_dump({
            "benchmark_root": str(benchmark_root),
            "bddl_files": str(benchmark_root / "bddl_files"),
            "init_states": str(benchmark_root / "init_files"),
            "datasets": str(home / "libero" / "datasets"),
            "assets": str(benchmark_root / "assets"),
        }), encoding="utf-8")


def allow_legacy_torch_load() -> None:
    """LIBERO init-state files require weights_only=False on newer PyTorch."""
    original = torch.load
    if getattr(original, "_feature_action_libero_compat", False):
        return

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    compatible_load._feature_action_libero_compat = True
    torch.load = compatible_load



def apply_video_flip(frame: np.ndarray, mode: str) -> np.ndarray:
    """Flip only the human-readable rollout video, never the policy input."""
    if mode == "none":
        return frame
    if mode == "vertical":
        return np.ascontiguousarray(frame[::-1, :, :])
    if mode == "horizontal":
        return np.ascontiguousarray(frame[:, ::-1, :])
    if mode == "both":
        return np.ascontiguousarray(frame[::-1, ::-1, :])
    raise ValueError(f"unsupported video_flip={mode!r}")


def display_frame(frame: np.ndarray, video_flip: str = "both") -> np.ndarray:
    """Return an RGB image for logging/video while preserving multi-view policy input."""
    if frame.ndim == 4:
        frame = np.concatenate([view for view in frame], axis=1)
    return apply_video_flip(frame, video_flip)

def save_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

def save_training_rollout(
    path: Path,
    *,
    camera_frames: dict[str, list[np.ndarray]],
    robot_states: list[np.ndarray],
    actions: list[np.ndarray],
    instruction: str,
    task_name: str,
    task_id: int,
    trial: int,
    success: bool,
    checkpoint: str,
) -> Path | None:
    """Persist one policy trajectory in the native LIBERO HDF5 schema.

    Each observation/state is the pre-action state paired with the action at
    the same index. Dummy wait controls are deliberately absent.
    """
    if not actions:
        return None
    length = len(actions)
    if len(robot_states) != length:
        raise ValueError(
            f"rollout robot-state/action length mismatch: {len(robot_states)} vs {length}"
        )
    for camera, frames in camera_frames.items():
        if len(frames) != length:
            raise ValueError(
                f"rollout {camera} frame/action length mismatch: {len(frames)} vs {length}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as h5:
        data = h5.require_group("data")
        data.attrs["problem_info"] = json.dumps(
            {"language_instruction": instruction, "task_name": task_name}
        )
        name = f"demo_{trial}"
        if name in data:
            del data[name]
        demo = data.create_group(name)
        demo.attrs["source"] = "policy_rollout"
        demo.attrs["success"] = np.bool_(success)
        demo.attrs["task_name"] = task_name
        demo.attrs["task_id"] = np.int64(task_id)
        demo.attrs["trial"] = np.int64(trial)
        demo.attrs["checkpoint"] = checkpoint
        demo.create_dataset(
            "actions", data=np.asarray(actions, dtype=np.float32), compression="lzf"
        )
        demo.create_dataset(
            "robot_states",
            data=np.asarray(robot_states, dtype=np.float32),
            compression="lzf",
        )
        obs = demo.create_group("obs")
        for camera, frames in camera_frames.items():
            obs.create_dataset(
                camera,
                data=np.asarray(frames, dtype=np.uint8),
                compression="lzf",
            )
    return path



def _feature_to_pca_img(feats: torch.Tensor, frame_size: int) -> np.ndarray:
    """Render one VFM spatial feature map [N,D] as a compact PCA RGB image."""
    feats = torch.nan_to_num(feats.detach().float().cpu())
    side = int(round(float(feats.shape[0]) ** 0.5))
    if side * side != feats.shape[0]:
        # Fallback: draw a 1D token strip if the feature count is not square.
        side = 1
        grid = feats.reshape(1, feats.shape[0], -1)
    else:
        grid = feats.reshape(side, side, -1)
    flat = grid.reshape(-1, grid.shape[-1])
    flat = flat - flat.mean(0, keepdim=True)
    if flat.shape[0] < 3:
        y = flat[:, :3] if flat.shape[-1] >= 3 else torch.nn.functional.pad(flat, (0, 3 - flat.shape[-1]))
    else:
        _, _, vh = torch.linalg.svd(flat, full_matrices=False)
        basis = vh[:3].T
        y = flat @ basis
    y = y.reshape(grid.shape[0], grid.shape[1], 3)
    y = y - y.amin((0, 1), keepdim=True)
    y = y / y.amax((0, 1), keepdim=True).clamp(min=1e-6)
    img = (y.numpy() * 255).round().astype(np.uint8)
    return cv2.resize(img, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def _decode_future_vfm(policy: EvalFeatureConditionedActionPolicy, observed_frames: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor | None:
    """Decode predicted DeltaTok future tokens to VFM spatial maps for visualization.

    Returns [L,V,N,D] for multi-view or [L,1,N,D] for single-view.  This is only
    a probe path; action inference still attends to the original tokens.
    """
    tokenizer = policy.network.world_rollout.delta_world.tokenizer
    if not hasattr(tokenizer, "decode"):
        return None
    try:
        if observed_frames.ndim == 6:
            # [B,T,V,C,H,W] with B=1 -> flatten views for DeltaTok decoder.
            batch, ctx, views = observed_frames.shape[:3]
            if batch != 1:
                return None
            flat_frames = observed_frames.permute(0, 2, 1, 3, 4, 5).reshape(
                batch * views, ctx, *observed_frames.shape[3:]
            )
            rope = tokenizer._rope(flat_frames)
            current = policy.network.world_rollout.encode_observations(observed_frames).vfm_features[:, :, -1]
            x = current.reshape(batch * views, current.shape[-2], current.shape[-1])
            ft = future_tokens[0].permute(1, 0, 2).reshape(views, future_tokens.shape[1], -1)
            decoded = []
            for step in range(ft.shape[1]):
                z = ft[:, step].to(device=x.device, dtype=x.dtype).unsqueeze(1)
                x = tokenizer.decode(z, x, rope)
                decoded.append(x.reshape(batch, views, x.shape[-2], x.shape[-1])[0].detach().float().cpu())
            return torch.stack(decoded, dim=0)
        if observed_frames.ndim == 5:
            batch = observed_frames.shape[0]
            if batch != 1:
                return None
            rope = tokenizer._rope(observed_frames)
            current = policy.network.world_rollout.encode_observations(observed_frames).vfm_features[:, -1]
            x = current
            decoded = []
            for step in range(future_tokens.shape[1]):
                z = future_tokens[:, step].to(device=x.device, dtype=x.dtype).unsqueeze(1)
                x = tokenizer.decode(z, x, rope)
                decoded.append(x[0:1].detach().float().cpu())
            return torch.stack(decoded, dim=0).permute(0, 1, 2, 3)
    except Exception as error:  # Keep evaluation alive if visualization decoding fails.
        LOG.warning("failed to decode world future tokens for probe: %s", error)
        return None
    return None


def save_world_probe(
    path: Path,
    policy: EvalFeatureConditionedActionPolicy,
    observed_frames: torch.Tensor,
    condition: Any,
    *,
    task_id: int,
    trial: int,
    step: int,
    instruction: str,
    video_flip: str,
    future_step_stride: int,
    save_visualization: bool = True,
    metadata: dict[str, Any] | None = None,
) -> PendingWorldProbe:
    """Save online DeltaWorld condition diagnostics for one replan."""
    path.mkdir(parents=True, exist_ok=True)
    frame_size = int(observed_frames.shape[-1])
    future = condition.future_tokens.detach().float().cpu()
    anchor = condition.current_vfm_features.detach().float().cpu()
    selected = condition.selected_branch.detach().cpu().tolist()
    mask = condition.future_mask.detach().cpu().bool()
    decoded = (
        _decode_future_vfm(policy, observed_frames, condition.future_tokens)
        if save_visualization else None
    )
    pred_current_vfm_cosine = decoded_future_current_vfm_cosine(decoded, anchor)
    future_len = int(future.shape[1])
    sheet_path: Path | None = None
    if save_visualization:
        # Convert context RGB to [T,V,H,W,3] for display.
        obs_cpu = observed_frames.detach().cpu()
        if obs_cpu.ndim == 6:
            rgb = obs_cpu[0].permute(0, 1, 3, 4, 2).numpy().astype(np.uint8)
        else:
            rgb = obs_cpu[0].permute(0, 2, 3, 1).numpy().astype(np.uint8)[:, None]
        context_len, views = rgb.shape[:2]
        rows: list[tuple[str, list[np.ndarray]]] = []
        for view in range(views):
            rows.append((
                f"view{view}_rgb",
                [apply_video_flip(rgb[t, view], video_flip) for t in range(context_len)]
                + [np.ones_like(rgb[0, view]) * 255 for _ in range(future_len)],
            ))
            anchor_feats = anchor[0, view] if anchor.ndim == 4 else anchor[0]
            rows.append((
                f"view{view}_anchor_pca",
                [np.ones_like(rgb[0, view]) * 255 for _ in range(context_len - 1)]
                + [_feature_to_pca_img(anchor_feats, frame_size)]
                + [np.ones_like(rgb[0, view]) * 255 for _ in range(future_len)],
            ))
            if decoded is not None:
                pred_imgs = [np.ones_like(rgb[0, view]) * 255 for _ in range(context_len)]
                for t in range(future_len):
                    pred_imgs.append(_feature_to_pca_img(decoded[t, view], frame_size))
                rows.append((f"view{view}_decoded_future_pca", pred_imgs))

        cell = 128
        label_w = 190
        font_h = 22
        cols = context_len + future_len
        canvas = Image.new(
            "RGB", (label_w + cell * cols, (cell + font_h) * len(rows)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        labels = [f"ctx{i}" for i in range(context_len)] + [f"f{i+1}" for i in range(future_len)]
        for r, (label, imgs) in enumerate(rows):
            y = r * (cell + font_h)
            draw.text((4, y + 4), label, fill=(0, 0, 0))
            for c, img in enumerate(imgs):
                if r == 0:
                    draw.text((label_w + c * cell + 4, y + 4), labels[c], fill=(255, 0, 0))
                pil = Image.fromarray(img).resize((cell, cell), Image.BILINEAR)
                canvas.paste(pil, (label_w + c * cell, y + font_h))
        sheet_path = path / f"task{task_id:02d}_trial{trial:02d}_step{step:04d}_world_probe.jpg"
        canvas.save(sheet_path)

    future_flat = future.flatten(2)
    token_std = future_flat.std(dim=-1, unbiased=False)[0].tolist()
    adjacent_cos = []
    if future_flat.shape[1] > 1:
        adjacent_cos = torch.nn.functional.cosine_similarity(
            future_flat[:, :-1], future_flat[:, 1:], dim=-1
        )[0].tolist()
    expected_steps = tuple(
        step + (future_index + 1) * int(future_step_stride)
        for future_index in range(future_len)
    )
    record = {
        "task_id": task_id,
        "trial": trial,
        "step": step,
        "instruction": instruction,
        "selected_branch": selected,
        "context_tokens": int(observed_frames.shape[1]),
        "future_shape": list(future.shape),
        "anchor_shape": list(anchor.shape),
        "future_mask_true": int(mask.sum().item()),
        "future_token_std_by_step": token_std,
        "future_adjacent_cosine": adjacent_cos,
        "pred_current_vfm_cosine_by_step": pred_current_vfm_cosine,
        "gt_source": "closed_loop_rollout_observation",
        "gt_status": "pending",
        "gt_expected_rollout_steps": list(expected_steps),
        "gt_matched_rollout_steps": [],
        "pred_gt_token_cosine_by_step": None,
        "pred_gt_vfm_cosine_by_step": None,
        "pred_gt_vfm_mse_by_step": None,
        "sheet": str(sheet_path) if sheet_path is not None else None,
        "decoded_future_pca": decoded is not None,
    }
    prediction_phase = (metadata or {}).get("world_consistency_prediction_phase")
    if isinstance(prediction_phase, dict):
        record["world_consistency_prediction_phase"] = prediction_phase
    json_path = path / f"task{task_id:02d}_trial{trial:02d}_step{step:04d}_world_probe.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return PendingWorldProbe(
        json_path=json_path,
        record=record,
        future_tokens=future,
        predicted_vfm=decoded,
        current_vfm_features=anchor,
        expected_steps=expected_steps,
        observed_frames={},
    )


def decoded_future_current_vfm_cosine(
    decoded_future_vfm: torch.Tensor | None,
    current_vfm_features: torch.Tensor,
) -> list[float] | None:
    """Compare every decoded predicted DINO map with the fixed current anchor.

    ``decoded_future_vfm`` is ``[L,V,N,D]``. The current-frame anchor accepts
    either multi-view ``[B,V,N,D]`` or single-view ``[B,N,D]``. Views, patches,
    and channels are reduced jointly, leaving one cosine value per future step.
    """
    if decoded_future_vfm is None:
        return None
    decoded = torch.nan_to_num(decoded_future_vfm.detach().float())
    anchor = torch.nan_to_num(current_vfm_features.detach().float())
    if anchor.ndim == 4:
        anchor = anchor[0]
    elif anchor.ndim == 3:
        anchor = anchor[0].unsqueeze(0)
    else:
        raise ValueError(
            "current VFM anchor must be [B,V,N,D] or [B,N,D], got "
            f"{tuple(current_vfm_features.shape)}"
        )
    if decoded.ndim != 4 or decoded.shape[1:] != anchor.shape:
        raise ValueError(
            "decoded future/current VFM shapes differ after removing time: "
            f"{tuple(decoded.shape)} vs {tuple(anchor.shape)}"
        )
    current = anchor.unsqueeze(0).expand(decoded.shape[0], *anchor.shape)
    return torch.nn.functional.cosine_similarity(
        decoded.flatten(1), current.flatten(1), dim=-1
    ).tolist()


def compute_world_probe_gt_metrics(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    predicted_vfm: torch.Tensor | None,
    target_vfm: torch.Tensor,
) -> dict[str, list[float] | None]:
    """Compare one predicted future with aligned closed-loop observations.

    All tensors have time as their first dimension. Camera, patch, and channel
    axes are reduced jointly, yielding one scalar per matched future step.
    """
    if predicted_tokens.shape != target_tokens.shape:
        raise ValueError(
            "predicted/target future-token shapes differ: "
            f"{tuple(predicted_tokens.shape)} vs {tuple(target_tokens.shape)}"
        )
    pred_token_flat = torch.nan_to_num(predicted_tokens.float()).flatten(1)
    target_token_flat = torch.nan_to_num(target_tokens.float()).flatten(1)
    token_cosine = torch.nn.functional.cosine_similarity(
        pred_token_flat, target_token_flat, dim=-1
    )

    vfm_cosine = None
    vfm_mse = None
    if predicted_vfm is not None:
        if predicted_vfm.shape != target_vfm.shape:
            raise ValueError(
                "predicted/target VFM shapes differ: "
                f"{tuple(predicted_vfm.shape)} vs {tuple(target_vfm.shape)}"
            )
        pred_vfm_flat = torch.nan_to_num(predicted_vfm.float()).flatten(1)
        target_vfm_flat = torch.nan_to_num(target_vfm.float()).flatten(1)
        vfm_cosine = torch.nn.functional.cosine_similarity(
            pred_vfm_flat, target_vfm_flat, dim=-1
        ).tolist()
        vfm_mse = ((pred_vfm_flat - target_vfm_flat) ** 2).mean(dim=-1).tolist()

    return {
        "pred_gt_token_cosine_by_step": token_cosine.tolist(),
        "pred_gt_vfm_cosine_by_step": vfm_cosine,
        "pred_gt_vfm_mse_by_step": vfm_mse,
    }


@torch.no_grad()
def _encode_world_probe_gt(
    policy: EvalFeatureConditionedActionPolicy,
    frames: list[np.ndarray],
    current_vfm_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode rollout RGB into DeltaTok targets and absolute DINO features."""
    array = np.stack(frames, axis=0)
    if array.ndim == 5:  # [L,V,H,W,3]
        future_frames = torch.from_numpy(np.ascontiguousarray(array)).permute(
            0, 1, 4, 2, 3
        ).unsqueeze(0)
    elif array.ndim == 4:  # [L,H,W,3]
        future_frames = torch.from_numpy(np.ascontiguousarray(array)).permute(
            0, 3, 1, 2
        ).unsqueeze(0)
    else:
        raise ValueError(f"unexpected rollout future RGB shape {array.shape}")

    rollout = policy.network.world_rollout
    device = next(rollout.delta_world.parameters()).device
    future_frames = future_frames.to(device)
    anchor = current_vfm_features.to(device)
    tokenizer = rollout.delta_world.tokenizer
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        if future_frames.ndim == 6:
            batch, future_len, views = future_frames.shape[:3]
            flat_frames = future_frames.permute(0, 2, 1, 3, 4, 5).reshape(
                batch * views, future_len, *future_frames.shape[3:]
            )
            current_flat = anchor.reshape(
                batch * views, anchor.shape[-2], anchor.shape[-1]
            )
            flat_vfm = tokenizer.backbone(flat_frames)
            rope = tokenizer._rope(flat_frames)
            flat_tokens = tokenizer.tokenize_offline(current_flat, flat_vfm, rope)
            target_tokens = flat_tokens.squeeze(2).reshape(
                batch, views, future_len, flat_vfm.shape[-1]
            ).permute(0, 2, 1, 3)
            target_vfm = flat_vfm.reshape(
                batch, views, future_len, flat_vfm.shape[-2], flat_vfm.shape[-1]
            ).permute(0, 2, 1, 3, 4)
        else:
            raw_vfm = tokenizer.backbone(future_frames)
            rope = tokenizer._rope(future_frames)
            target_tokens = tokenizer.tokenize_offline(anchor, raw_vfm, rope).squeeze(2)
            target_vfm = raw_vfm.unsqueeze(2)
    return target_tokens.detach().float().cpu(), target_vfm.detach().float().cpu()


def evaluate_world_probe_prefix_token_cosine(
    probe: PendingWorldProbe,
    policy: EvalFeatureConditionedActionPolicy,
    *,
    horizon: int,
) -> dict[str, Any] | None:
    """Evaluate one future-token horizon as soon as its observation arrives."""
    if horizon <= 0 or horizon > len(probe.expected_steps):
        return None
    if horizon in probe.evaluated_prefix_horizons:
        return None
    prefix_steps = probe.expected_steps[:horizon]
    if not all(step in probe.observed_frames for step in prefix_steps):
        return None

    frames = [probe.observed_frames[step] for step in prefix_steps]
    try:
        target_tokens, _ = _encode_world_probe_gt(
            policy, frames, probe.current_vfm_features
        )
        predicted_tokens = probe.future_tokens[0, :horizon]
        if predicted_tokens.ndim == 2:
            predicted_tokens = predicted_tokens.unsqueeze(1)
        target_tokens = target_tokens[0, :horizon]
        if target_tokens.ndim == 2:
            target_tokens = target_tokens.unsqueeze(1)
        values = torch.nn.functional.cosine_similarity(
            torch.nan_to_num(predicted_tokens.float()).flatten(1),
            torch.nan_to_num(target_tokens.float()).flatten(1),
            dim=-1,
        ).tolist()
        event = {
            "prediction_step": int(probe.record["step"]),
            "observed_step": int(prefix_steps[-1]),
            "horizon": int(horizon),
            "pred_gt_token_cosine": float(values[-1]),
        }
        prediction_phase = probe.record.get("world_consistency_prediction_phase")
        if isinstance(prediction_phase, dict):
            event["prediction_goal_bowl_eef_distance"] = prediction_phase.get(
                "goal_bowl_eef_distance"
            )
        probe.record.setdefault("online_prefix_checks", []).append(event)
        probe.evaluated_prefix_horizons.add(horizon)
        probe.json_path.write_text(
            json.dumps(probe.record, indent=2), encoding="utf-8"
        )
        return event
    except Exception as error:
        LOG.warning(
            "failed to compute online f%d token consistency for %s: %s",
            horizon, probe.json_path, error,
        )
        probe.evaluated_prefix_horizons.add(horizon)
        return {
            "prediction_step": int(probe.record["step"]),
            "observed_step": int(prefix_steps[-1]),
            "horizon": int(horizon),
            "pred_gt_token_cosine": None,
            "error": str(error),
        }


def finalize_world_probe_gt(
    probe: PendingWorldProbe,
    policy: EvalFeatureConditionedActionPolicy,
) -> dict[str, Any]:
    """Write all currently available, temporally aligned online-GT metrics."""
    matched_steps: list[int] = []
    frames: list[np.ndarray] = []
    # Only use the longest contiguous prefix. This preserves DeltaTok's
    # autoregressive target order when an episode ends before the full horizon.
    for expected_step in probe.expected_steps:
        if expected_step not in probe.observed_frames:
            break
        matched_steps.append(expected_step)
        frames.append(probe.observed_frames[expected_step])

    record = probe.record
    record["gt_matched_rollout_steps"] = matched_steps
    if not frames:
        record["gt_status"] = "unavailable"
        probe.json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    try:
        target_tokens, target_vfm = _encode_world_probe_gt(
            policy, frames, probe.current_vfm_features
        )
        count = len(frames)
        predicted_tokens = probe.future_tokens[0, :count]
        if predicted_tokens.ndim == 2:
            predicted_tokens = predicted_tokens.unsqueeze(1)
        target_tokens = target_tokens[0, :count]
        if target_tokens.ndim == 2:
            target_tokens = target_tokens.unsqueeze(1)
        predicted_vfm = (
            probe.predicted_vfm[:count]
            if probe.predicted_vfm is not None else None
        )
        target_vfm = target_vfm[0, :count]
        record.update(compute_world_probe_gt_metrics(
            predicted_tokens, target_tokens, predicted_vfm, target_vfm
        ))
        record["gt_status"] = (
            "complete" if count == len(probe.expected_steps) else "partial"
        )
    except Exception as error:
        LOG.warning("failed to compute online world-probe GT metrics: %s", error)
        record["gt_status"] = "error"
        record["gt_error"] = str(error)
    probe.json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def update_pending_world_probes(
    pending: list[PendingWorldProbe],
    completed: list[dict[str, Any]],
    policy: EvalFeatureConditionedActionPolicy,
    *,
    observed_step: int,
    frame: np.ndarray,
    online_consistency_horizon: int | None = None,
) -> list[dict[str, Any]]:
    """Attach one rollout observation and settle completed predictions."""
    online_events: list[dict[str, Any]] = []
    for probe in list(pending):
        if observed_step in probe.expected_steps:
            probe.observed_frames[observed_step] = np.ascontiguousarray(frame).copy()
        if online_consistency_horizon is not None:
            event = evaluate_world_probe_prefix_token_cosine(
                probe, policy, horizon=online_consistency_horizon
            )
            if event is not None:
                online_events.append(event)
        if all(step in probe.observed_frames for step in probe.expected_steps):
            completed.append(finalize_world_probe_gt(probe, policy))
            pending.remove(probe)
    return online_events


def summarize_world_probe_gt(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-step online-GT metrics over one rollout episode."""
    metric_names = (
        "pred_current_vfm_cosine_by_step",
        "pred_gt_token_cosine_by_step",
        "pred_gt_vfm_cosine_by_step",
        "pred_gt_vfm_mse_by_step",
    )
    summary: dict[str, Any] = {
        "num_probes": len(records),
        "complete_probes": sum(record.get("gt_status") == "complete" for record in records),
        "partial_probes": sum(record.get("gt_status") == "partial" for record in records),
        "unavailable_or_error_probes": sum(
            record.get("gt_status") in {"unavailable", "error"} for record in records
        ),
    }
    for name in metric_names:
        values = [
            float(value)
            for record in records
            for value in (record.get(name) or [])
        ]
        summary[f"{name.removesuffix('_by_step')}_mean"] = (
            float(np.mean(values)) if values else None
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--libero-home", type=Path, default=os.environ.get("LIBERO_HOME"))
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument(
        "--benchmark-family", choices=("libero", "libero-plus", "libero-pro"),
        default="libero",
    )
    parser.add_argument("--benchmark-base-suite", type=str)
    parser.add_argument("--benchmark-perturbation", type=str)
    parser.add_argument(
        "--instruction-source", choices=("task", "bddl"), default="task",
        help="Use benchmark task metadata or the :language field parsed from BDDL.",
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument(
        "--task-ids",
        type=str,
        help="Comma-separated task ids evaluated in one model load, e.g. 2,16,40.",
    )
    parser.add_argument("--random-task", action="store_true", help="Evaluate one random task from the selected suite.")
    parser.add_argument("--random-task-seed", type=int, help="Seed for --random-task; defaults to --seed.")
    parser.add_argument(
        "--libero-plus-classification",
        type=Path,
        help=(
            "Attach the official LIBERO-Plus classification and evaluate the "
            "complete suite (or complete --libero-plus-category) without subsampling."
        ),
    )
    parser.add_argument(
        "--libero-plus-balanced-classification",
        type=Path,
        help="Select a deterministic balanced-fast subset using LIBERO-Plus task_classification.json.",
    )
    parser.add_argument(
        "--libero-plus-balanced-tasks-per-category",
        type=int,
        default=10,
        help="Balanced-fast tasks per perturbation category (default: 10, or 70 tasks per suite).",
    )
    parser.add_argument("--libero-plus-balanced-seed", type=int, default=0)
    parser.add_argument(
        "--libero-plus-category",
        choices=LIBERO_PLUS_CATEGORIES,
        help=(
            "With --libero-plus-classification, evaluate every instance in one "
            "official category. With the balanced selector, use its diagnostic subset."
        ),
    )
    parser.add_argument(
        "--world-prediction-steps-override",
        type=int,
        help="Temporarily override model.init_args.world_prediction_steps during eval, e.g. 6.",
    )
    parser.add_argument(
        "--world-checkpoint-override",
        type=Path,
        help=(
            "Evaluation-only frozen DeltaWorld checkpoint override. ActionDiT "
            "weights continue to come from --checkpoint unchanged."
        ),
    )
    parser.add_argument(
        "--instruction-override",
        type=str,
        help="Override LIBERO task.language for language-grounding ablations. The simulator task/BDDL is unchanged.",
    )
    parser.add_argument(
        "--disable-world-future-tokens",
        action="store_true",
        help=(
            "Ablation only: keep current-frame spatial anchors and native "
            "text/proprio/history condition, but remove predicted DeltaWorld "
            "future tokens from the ActionDiT K/V cache."
        ),
    )
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--trial-start", type=int, default=0, help="First fixed initial-state/trial index (for diagnostic resume).")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Episode policy-step limit. Defaults to the LIBERO/LeRobot suite "
            "length: spatial=220, object=280, goal=300, libero_90=400, "
            "libero_10=520."
        ),
    )
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument(
        "--disable-action-history",
        action="store_true",
        help=(
            "Ablation only: keep the loaded ActionDiT architecture unchanged, "
            "but mask out all executed-action history tokens during rollout."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="Fuse overlapping action chunks at matching absolute control steps.",
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.1,
        help="Exponential age decay; larger values favor the newest chunk more strongly.",
    )
    parser.add_argument(
        "--temporal-disagreement-aware",
        action="store_true",
        help="Gate stale chunk weights when aligned action plans disagree.",
    )
    parser.add_argument(
        "--temporal-disagreement-window",
        type=int,
        default=4,
        help="Number of aligned current/future actions used to compare chunks.",
    )
    parser.add_argument("--temporal-position-low", type=float, default=0.08)
    parser.add_argument("--temporal-position-high", type=float, default=0.20)
    parser.add_argument("--temporal-rotation-low", type=float, default=0.10)
    parser.add_argument("--temporal-rotation-high", type=float, default=0.30)
    parser.add_argument(
        "--temporal-direction-cosine-threshold",
        type=float,
        default=0.0,
        help="Use only the newest chunk when aligned motion cosine falls below this value.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/feature_action_libero")
    parser.add_argument(
        "--resume-output-dir",
        type=Path,
        help=(
            "Stable run leaf used for crash-safe evaluation. Completed episodes "
            "are restored from results.partial.json and skipped."
        ),
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        help="Optional suffix for the timestamped output directory, useful for ablations.",
    )
    parser.add_argument(
        "--video-flip", choices=("none", "vertical", "horizontal", "both"),
        default="both",
        help="Flip saved rollout videos for human display only; policy input is unchanged.",
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--paper-record", action="store_true",
                        help="Record actual condition tokens, native attention, and aligned observations for paper analysis.")
    parser.add_argument(
        "--save-training-rollouts",
        action="store_true",
        help=(
            "Save policy-phase pre-action RGB/proprio/action trajectories as "
            "LIBERO-compatible HDF5 files for action-conditioned world-model training."
        ),
    )
    parser.add_argument(
        "--save-world-probe",
        action="store_true",
        help="Save online DeltaWorld condition diagnostics at replan steps.",
    )
    parser.add_argument(
        "--world-probe-every-replan",
        type=int,
        default=1,
        help="Save one DeltaWorld probe every N replans when --save-world-probe is set.",
    )
    parser.add_argument(
        "--task9-world-consistency-replan",
        action="store_true",
        help=(
            "LIBERO-Spatial task-9 pilot: evaluate predicted-vs-observed future-token "
            "cosine online and discard the current action plan when the score is low, "
            "but only before approaching/grasping the goal bowl."
        ),
    )
    parser.add_argument(
        "--world-consistency-horizon",
        type=int,
        default=4,
        help="Future-token index used by the online consistency trigger (default: f4).",
    )
    parser.add_argument(
        "--world-consistency-token-cosine-threshold",
        type=float,
        default=0.94,
        help=(
            "Soft f-horizon cosine threshold. A score below this triggers only "
            "when goal-distance progress is also insufficient."
        ),
    )
    parser.add_argument(
        "--world-consistency-hard-token-cosine-threshold",
        type=float,
        default=0.925,
        help="Severe inconsistency threshold that bypasses the progress requirement.",
    )
    parser.add_argument(
        "--world-consistency-far-distance",
        type=float,
        default=0.12,
        help="Enable the task-9 trigger only while goal-bowl/eef distance exceeds this many metres.",
    )
    parser.add_argument(
        "--world-consistency-min-progress",
        type=float,
        default=0.015,
        help=(
            "Required decrease in goal-bowl/eef distance over the checked future "
            "horizon; soft-low scores trigger only below this progress (metres)."
        ),
    )
    parser.add_argument(
        "--world-consistency-cooldown-steps",
        type=int,
        default=16,
        help="Minimum control steps between consistency-triggered replans.",
    )
    parser.add_argument(
        "--world-consistency-max-triggers",
        type=int,
        default=3,
        help="Maximum consistency-triggered replans in one episode.",
    )
    parser.add_argument(
        "--world-consistency-max-policy-step",
        type=int,
        default=128,
        help="Disable consistency-triggered replans after this policy step.",
    )
    parser.add_argument(
        "--world-consistency-retry-candidates",
        type=int,
        default=1,
        help=(
            "Number of stochastic world/action plans generated after a trigger; "
            "the action-space consensus medoid is selected. Default 1 keeps this "
            "costly experimental ablation disabled."
        ),
    )
    parser.add_argument(
        "--world-consistency-retry-lookahead",
        type=int,
        default=4,
        help="Leading action steps used for retry-plan consensus selection.",
    )
    parser.add_argument(
        "--save-success-diagnostics",
        action="store_true",
        help="Save per-step LIBERO goal-object/contact/geometry/gripper diagnostics.",
    )
    parser.add_argument(
        "--oracle-adaptive-replan",
        action="store_true",
        help="Use privileged simulator geometry to choose far/place replan length for ablations.",
    )
    parser.add_argument(
        "--oracle-one-way-switch",
        action="store_true",
        help="With oracle adaptive replan, switch permanently to place replan length after first place-phase detection.",
    )
    parser.add_argument("--oracle-far-replan-steps", type=int, default=8)
    parser.add_argument("--oracle-place-replan-steps", type=int, default=12)
    parser.add_argument("--oracle-place-xy-threshold", type=float, default=0.10)
    parser.add_argument(
        "--oracle-no-require-active-goal",
        action="store_true",
        help="Do not require the active manipulated bowl to be the goal bowl before using place replan length.",
    )
    parser.add_argument(
        "--oracle-no-require-grasped",
        action="store_true",
        help="Do not require the goal bowl to be grasped before using place replan length.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.world_probe_every_replan <= 0:
        raise ValueError("--world-probe-every-replan must be positive")
    if args.world_consistency_horizon <= 0:
        raise ValueError("--world-consistency-horizon must be positive")
    if not -1 <= args.world_consistency_token_cosine_threshold <= 1:
        raise ValueError("--world-consistency-token-cosine-threshold must be in [-1,1]")
    if not -1 <= args.world_consistency_hard_token_cosine_threshold <= 1:
        raise ValueError(
            "--world-consistency-hard-token-cosine-threshold must be in [-1,1]"
        )
    if (
        args.world_consistency_hard_token_cosine_threshold
        > args.world_consistency_token_cosine_threshold
    ):
        raise ValueError("the hard consistency threshold must not exceed the soft threshold")
    if args.world_consistency_far_distance <= 0:
        raise ValueError("--world-consistency-far-distance must be positive")
    if args.world_consistency_min_progress < 0:
        raise ValueError("--world-consistency-min-progress must be non-negative")
    if args.world_consistency_cooldown_steps < 0:
        raise ValueError("--world-consistency-cooldown-steps must be non-negative")
    if args.world_consistency_max_triggers <= 0:
        raise ValueError("--world-consistency-max-triggers must be positive")
    if args.world_consistency_max_policy_step <= 0:
        raise ValueError("--world-consistency-max-policy-step must be positive")
    if args.world_consistency_retry_candidates <= 0:
        raise ValueError("--world-consistency-retry-candidates must be positive")
    if args.world_consistency_retry_lookahead <= 0:
        raise ValueError("--world-consistency-retry-lookahead must be positive")
    if args.oracle_far_replan_steps <= 0 or args.oracle_place_replan_steps <= 0:
        raise ValueError("oracle replan step values must be positive")
    if args.oracle_place_xy_threshold <= 0:
        raise ValueError("--oracle-place-xy-threshold must be positive")
    if args.replan_steps <= 0:
        raise ValueError("--replan-steps must be positive")
    if args.temporal_ensemble_decay < 0:
        raise ValueError("--temporal-ensemble-decay must be non-negative")
    if args.temporal_disagreement_aware and not args.temporal_ensemble:
        raise ValueError("--temporal-disagreement-aware requires --temporal-ensemble")
    if args.task9_world_consistency_replan and args.oracle_adaptive_replan:
        raise ValueError(
            "--task9-world-consistency-replan and --oracle-adaptive-replan "
            "cannot be enabled together"
        )
    if args.temporal_disagreement_window <= 0:
        raise ValueError("--temporal-disagreement-window must be positive")
    if not 0 <= args.temporal_position_low < args.temporal_position_high:
        raise ValueError("temporal position thresholds must satisfy 0 <= low < high")
    if not 0 <= args.temporal_rotation_low < args.temporal_rotation_high:
        raise ValueError("temporal rotation thresholds must satisfy 0 <= low < high")
    if not -1 <= args.temporal_direction_cosine_threshold <= 1:
        raise ValueError("temporal direction cosine threshold must be in [-1,1]")

    configure_libero(args.libero_home)
    allow_legacy_torch_load()
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as error:
        raise RuntimeError("LIBERO is not importable; install it or pass --libero-home") from error

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy, config = load_policy(
        args.config, args.checkpoint, device,
        world_prediction_steps_override=args.world_prediction_steps_override,
        world_checkpoint_override=args.world_checkpoint_override,
    )
    data = dataset_args(config)
    frame_size = config["data"]["init_args"].get("frame_size", 512)
    adapter = LiberoObservationAdapter(
        data.get("cameras", ["agentview_rgb"]), data.get("camera_layout", "first"), frame_size
    )
    context_frames = int(data.get("context_frames", 4))
    context_stride = int(data.get("context_frame_stride", 2))
    world_frame_stride = int(data.get("world_frame_stride", context_stride))
    action_horizon = int(data.get("action_horizon", 16))
    if args.task9_world_consistency_replan:
        effective_suite = args.benchmark_base_suite or args.suite
        if effective_suite != "libero_spatial":
            raise ValueError(
                "--task9-world-consistency-replan is currently restricted to "
                "LIBERO-Spatial task 9"
            )
        if args.world_consistency_horizon > policy.network.world_prediction_steps:
            raise ValueError(
                "--world-consistency-horizon exceeds the model world prediction "
                f"horizon ({policy.network.world_prediction_steps})"
            )
        trigger_delay = args.world_consistency_horizon * world_frame_stride
        if args.replan_steps <= trigger_delay:
            LOG.warning(
                "task-9 consistency f%d arrives after %d control steps, but "
                "--replan-steps=%d; the trigger cannot replan earlier than the "
                "normal schedule. Use --replan-steps > %d (recommended: 16).",
                args.world_consistency_horizon, trigger_delay,
                args.replan_steps, trigger_delay,
            )
    action_history_steps = int(policy.network.action_history_steps)
    dataset_history_steps = int(data.get("action_history_steps", 0))
    if dataset_history_steps != action_history_steps:
        raise ValueError(
            "model/dataset action_history_steps mismatch: "
            f"{action_history_steps} vs {dataset_history_steps}"
        )
    fps = float(data.get("fps", 20))
    history_len = (context_frames - 1) * context_stride + 1
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.max_steps is None:
        args.max_steps = int(LIBERO_SUITE_MAX_STEPS.get(args.benchmark_base_suite or args.suite, 400))
        LOG.info("using suite default max_steps=%d for suite=%s", args.max_steps, args.suite)
    if (
        args.libero_plus_classification is not None
        and args.libero_plus_balanced_classification is not None
    ):
        raise ValueError(
            "--libero-plus-classification and "
            "--libero-plus-balanced-classification are mutually exclusive"
        )
    category_is_selector = (
        args.libero_plus_category is not None
        and args.libero_plus_classification is not None
    )
    selectors = (
        int(args.random_task)
        + int(args.task_id is not None)
        + int(args.task_ids is not None)
        + int(args.libero_plus_balanced_classification is not None)
        + int(category_is_selector)
    )
    if selectors > 1:
        raise ValueError(
            "--random-task, --task-id, --task-ids, the balanced selector, and "
            "the official category selector are mutually exclusive"
        )
    if (
        args.libero_plus_category is not None
        and args.libero_plus_classification is None
        and args.libero_plus_balanced_classification is None
    ):
        raise ValueError("--libero-plus-category requires a classification selector")
    task_metadata: dict[int, dict[str, Any]] = {}
    task_selection = {"mode": "full-suite", "num_tasks": int(suite.n_tasks)}
    official_full_indices: list[int] | None = None
    official_full_selection: dict[str, Any] | None = None
    if args.libero_plus_classification is not None:
        official_full_indices, task_metadata, official_full_selection = (
            select_official_libero_plus_tasks(
                args.libero_plus_classification,
                args.suite,
                suite.get_task_names(),
            )
        )
    if args.random_task:
        rng = np.random.default_rng(args.random_task_seed if args.random_task_seed is not None else args.seed)
        selected_task = int(rng.integers(0, suite.n_tasks))
        task_indices = [selected_task]
        task_selection = {"mode": "random-task", "num_tasks": 1}
        LOG.info("randomly selected task_id=%d from suite=%s", selected_task, args.suite)
    elif args.task_ids is not None:
        try:
            task_indices = [int(value.strip()) for value in args.task_ids.split(",") if value.strip()]
        except ValueError as error:
            raise ValueError("--task-ids must be a comma-separated integer list") from error
        if not task_indices:
            raise ValueError("--task-ids cannot be empty")
        task_indices = list(dict.fromkeys(task_indices))
        invalid = [task_id for task_id in task_indices if not 0 <= task_id < suite.n_tasks]
        if invalid:
            raise ValueError(f"task ids outside [0,{suite.n_tasks}): {invalid}")
        task_selection = {"mode": "explicit-task-ids", "num_tasks": len(task_indices)}
    elif args.libero_plus_balanced_classification is not None:
        task_indices, task_metadata, task_selection = select_balanced_libero_plus_tasks(
            args.libero_plus_balanced_classification,
            args.suite,
            suite.get_task_names(),
            tasks_per_category=args.libero_plus_balanced_tasks_per_category,
            seed=args.libero_plus_balanced_seed,
            categories=(
                (args.libero_plus_category,)
                if args.libero_plus_category is not None
                else None
            ),
        )
        LOG.info(
            "selected LIBERO-Plus balanced-fast subset: suite=%s tasks=%d (%d/category)",
            args.suite,
            len(task_indices),
            args.libero_plus_balanced_tasks_per_category,
        )
    elif category_is_selector:
        task_indices, task_metadata, task_selection = select_official_libero_plus_tasks(
            args.libero_plus_classification,
            args.suite,
            suite.get_task_names(),
            categories=(args.libero_plus_category,),
        )
        LOG.info(
            "selected complete LIBERO-Plus category: suite=%s category=%s tasks=%d",
            args.suite,
            args.libero_plus_category,
            len(task_indices),
        )
    else:
        if args.task_id is not None:
            task_indices = [args.task_id]
            task_selection = {"mode": "explicit-task-id", "num_tasks": 1}
        elif official_full_indices is not None:
            task_indices = official_full_indices
            task_selection = official_full_selection
            LOG.info(
                "selected official full LIBERO-Plus suite: suite=%s tasks=%d",
                args.suite,
                len(task_indices),
            )
        else:
            task_indices = range(suite.n_tasks)
    if args.resume_output_dir is not None:
        output = args.resume_output_dir.resolve()
        run_name = output.name
    else:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        if args.run_tag:
            safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in args.run_tag)
            run_name = f"{run_name}_{safe_tag}"
        output = args.output_dir / args.checkpoint.stem / args.suite / run_name
    output.mkdir(parents=True, exist_ok=True)
    LOG.info("writing rollout outputs to %s", output)
    results = {
        "checkpoint": str(args.checkpoint),
        "action_checkpoint_world": policy.action_checkpoint_world,
        "effective_world_checkpoint": policy.eval_world_checkpoint,
        "world_checkpoint_overridden": bool(
            policy.world_checkpoint_overridden
        ),
        "suite": args.suite,
        "benchmark_family": args.benchmark_family,
        "benchmark_base_suite": args.benchmark_base_suite,
        "benchmark_perturbation": args.benchmark_perturbation,
        "instruction_source": args.instruction_source,
        "task_indices": list(task_indices),
        "task_selection": task_selection,
        "random_task": bool(args.random_task),
        "random_task_seed": args.random_task_seed if args.random_task_seed is not None else args.seed,
        "world_prediction_steps": int(policy.network.world_prediction_steps),
        "world_frame_stride": int(world_frame_stride),
        "action_history_steps": action_history_steps,
        "action_history_source": "actually_executed_policy_commands",
        "action_history_disabled": bool(args.disable_action_history),
        "world_future_tokens_disabled": bool(args.disable_world_future_tokens),
        "effective_world_future_tokens": (
            0 if args.disable_world_future_tokens else int(policy.network.world_prediction_steps)
        ),
        "warm_start_available_context": True,
        "temporal_ensemble": bool(args.temporal_ensemble),
        "temporal_ensemble_decay": float(args.temporal_ensemble_decay),
        "temporal_disagreement_aware": bool(args.temporal_disagreement_aware),
        "temporal_disagreement_window": int(args.temporal_disagreement_window),
        "temporal_position_thresholds": [args.temporal_position_low, args.temporal_position_high],
        "temporal_rotation_thresholds": [args.temporal_rotation_low, args.temporal_rotation_high],
        "temporal_direction_cosine_threshold": float(
            args.temporal_direction_cosine_threshold
        ),
        "save_success_diagnostics": bool(args.save_success_diagnostics),
        "save_training_rollouts": bool(args.save_training_rollouts),
        "oracle_adaptive_replan": bool(args.oracle_adaptive_replan),
        "oracle_one_way_switch": bool(args.oracle_one_way_switch),
        "oracle_far_replan_steps": int(args.oracle_far_replan_steps),
        "oracle_place_replan_steps": int(args.oracle_place_replan_steps),
        "oracle_place_xy_threshold": float(args.oracle_place_xy_threshold),
        "task9_world_consistency_replan": bool(args.task9_world_consistency_replan),
        "world_consistency_horizon": int(args.world_consistency_horizon),
        "world_consistency_token_cosine_threshold": float(
            args.world_consistency_token_cosine_threshold
        ),
        "world_consistency_hard_token_cosine_threshold": float(
            args.world_consistency_hard_token_cosine_threshold
        ),
        "world_consistency_far_distance": float(args.world_consistency_far_distance),
        "world_consistency_min_progress": float(args.world_consistency_min_progress),
        "world_consistency_cooldown_steps": int(
            args.world_consistency_cooldown_steps
        ),
        "world_consistency_max_triggers": int(args.world_consistency_max_triggers),
        "world_consistency_max_policy_step": int(
            args.world_consistency_max_policy_step
        ),
        "world_consistency_retry_candidates": int(
            args.world_consistency_retry_candidates
        ),
        "world_consistency_retry_lookahead": int(
            args.world_consistency_retry_lookahead
        ),
        "world_consistency_phase_gate": "pregrasp_goal_bowl_eef_distance",
        "instruction_override": args.instruction_override,
        "output_dir": str(output),
        "run_name": run_name,
        "seed": int(args.seed),
        "num_trials_per_task_requested": int(args.num_trials),
        "replan_steps": int(args.replan_steps),
        "num_inference_steps": int(args.num_inference_steps),
        "success_definition": "success_once_env_done",
        "tasks": {},
    }
    partial_path = output / "results.partial.json"
    if args.resume_output_dir is not None and partial_path.exists():
        expected = {
            key: results[key]
            for key in (
                "checkpoint", "suite", "benchmark_family",
                "benchmark_base_suite", "benchmark_perturbation", "task_indices",
                "seed", "num_trials_per_task_requested", "replan_steps",
                "num_inference_steps", "temporal_ensemble",
                "temporal_ensemble_decay",
            )
        }
        results = restore_partial_results(partial_path, expected)
        LOG.info(
            "resumed %d completed episodes from %s",
            int(results.get("trials", 0)), partial_path,
        )
    total_success = sum(
        int(bool(episode.get("success")))
        for task_result in results["tasks"].values()
        for episode in task_result.get("episodes", [])
    )
    total_trials = sum(
        len(task_result.get("episodes", []))
        for task_result in results["tasks"].values()
    )
    all_temporal_disagreement_records: list[dict[str, Any]] = []

    for suite_task_id in task_indices:
        task = suite.get_task(suite_task_id)
        task9_consistency_enabled = bool(
            args.task9_world_consistency_replan and suite_task_id == 9
        )
        if task9_consistency_enabled:
            LOG.info(
                "task=9 online world consistency enabled: f%d soft/hard "
                "cosine < %.6f/%.6f, min_progress=%.3fm, cooldown=%d, "
                "max_triggers=%d, max_policy_step=%d, phase=eef distance > "
                "%.3fm until first goal-bowl grasp",
                args.world_consistency_horizon,
                args.world_consistency_token_cosine_threshold,
                args.world_consistency_hard_token_cosine_threshold,
                args.world_consistency_min_progress,
                args.world_consistency_cooldown_steps,
                args.world_consistency_max_triggers,
                args.world_consistency_max_policy_step,
                args.world_consistency_far_distance,
            )
        task_instruction = str(task.language)
        states = suite.get_task_init_states(suite_task_id)
        if args.benchmark_family == "libero-pro" and len(states) < args.num_trials:
            raise RuntimeError(
                "LIBERO-Pro official evaluation requires the requested fixed "
                "initial states for every task, but "
                f"suite={args.suite} task={suite_task_id} has {len(states)} "
                f"states for --num-trials={args.num_trials}. Re-run "
                "scripts/setup_libero_pro.sh to prepare 50 states per task."
            )
        existing_task_result = results["tasks"].get(str(suite_task_id), {})
        records = list(existing_task_result.get("episodes", []))
        completed_trials = {int(record["trial"]) for record in records}
        num_task_trials = (
            args.num_trials
            if args.benchmark_family == "libero-pro"
            else min(args.num_trials, len(states))
        )
        if not 0 <= args.trial_start < num_task_trials:
            raise ValueError("trial-start must be in [0, num-trials)")
        requested_trials = set(range(args.trial_start, num_task_trials))
        if requested_trials.issubset(completed_trials):
            LOG.info(
                "task=%d already has all %d requested trials; skipping before env creation",
                suite_task_id, len(requested_trials),
            )
            continue
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        bddl_instruction = str(
            getattr(env, "language_instruction", task_instruction)
        )
        source_instruction = (
            bddl_instruction if args.instruction_source == "bddl" else task_instruction
        )
        instruction = (
            args.instruction_override
            if args.instruction_override is not None else source_instruction
        )
        if instruction != task_instruction:
            LOG.info(
                "task=%d instruction_source=%s task=%r bddl=%r selected=%r",
                suite_task_id, args.instruction_source, task_instruction,
                bddl_instruction, instruction,
            )
        cached_text_context, cached_text_mask, cached_world_condition = cache_instruction_conditions(
            policy, instruction
        )
        env.seed(args.seed)
        try:
            for trial in range(args.trial_start, num_task_trials):
                if trial in completed_trials:
                    LOG.info("task=%d trial=%d already complete; skipping", suite_task_id, trial)
                    continue
                env.reset()
                obs = env.set_init_state(states[trial])
                paper_recorder = None
                if args.paper_record:
                    from scripts.paper_visualization_recording import PaperRecorder
                    paper_recorder = PaperRecorder(
                        output / "paper_records" / f"task{suite_task_id:02d}_trial{trial:02d}.h5",
                        {"suite": args.suite, "task_id": suite_task_id, "trial": trial,
                         "instruction": instruction, "checkpoint": str(args.checkpoint),
                         "world_checkpoint": policy.eval_world_checkpoint,
                         "fps": fps, "wait_steps": args.wait_steps,
                         "world_frame_stride": world_frame_stride,
                         "world_prediction_steps": policy.network.world_prediction_steps,
                         "temporal_ensemble": args.temporal_ensemble,
                         "temporal_ensemble_decay": args.temporal_ensemble_decay,
                         "replan_steps": args.replan_steps, "seed": args.seed})
                history = deque(maxlen=history_len)
                executed_actions = deque(maxlen=max(action_history_steps, 1))
                pending, action_chunks, rendered, success = [], [], [], False
                rollout_camera_frames = {camera: [] for camera in adapter.cameras}
                rollout_robot_states: list[np.ndarray] = []
                rollout_actions: list[np.ndarray] = []
                temporal_disagreement_records: list[dict[str, Any]] = []
                pending_world_probes: list[PendingWorldProbe] = []
                completed_world_probe_records: list[dict[str, Any]] = []
                world_consistency_records: list[dict[str, Any]] = []
                world_consistency_disabled_after_grasp = False
                world_consistency_last_trigger_step: int | None = None
                world_consistency_trigger_count = 0
                world_consistency_retry_pending = False
                world_consistency_retry_records: list[dict[str, Any]] = []
                current_world_consistency_phase: dict[str, Any] | None = None
                if task9_consistency_enabled:
                    (
                        current_world_consistency_phase,
                        world_consistency_disabled_after_grasp,
                    ) = task9_world_consistency_phase(
                        env,
                        obs,
                        far_distance=args.world_consistency_far_distance,
                        disabled_after_grasp=False,
                    )
                diagnostic_records: list[dict[str, Any]] = []
                diagnostic_path = None
                diagnostic_summary_path = None
                diagnostic_stream = None
                diagnostic_initial_positions: dict[str, list[float]] = {}
                if args.save_success_diagnostics:
                    diagnostic_path = output / "success_diagnostics" / f"task{suite_task_id:02d}" / f"trial{trial:02d}.jsonl"
                    diagnostic_summary_path = diagnostic_path.with_name(f"trial{trial:02d}_summary.json")
                    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                    diagnostic_stream = diagnostic_path.open("w", encoding="utf-8")
                    diagnostic_initial_positions = _libero_object_positions(env)
                oracle_replan_records: list[dict[str, Any]] = []
                current_oracle_replan: dict[str, Any] | None = None
                oracle_long_mode = False
                oracle_switch_step = None
                replan_count = 0
                next_replan_policy_step = 0
                for step in range(args.max_steps + args.wait_steps):
                    frame = adapter.frame(obs)
                    history.append(frame)
                    rendered.append(display_frame(frame, args.video_flip))
                    if step < args.wait_steps:
                        action = DUMMY_ACTION
                    else:
                        policy_step = step - args.wait_steps
                        should_replan = (
                            not pending
                            if not args.temporal_ensemble
                            else not action_chunks or policy_step >= next_replan_policy_step
                        )
                        if should_replan:
                            probe_dir = None
                            save_scheduled_probe = (
                                args.save_world_probe
                                and replan_count % args.world_probe_every_replan == 0
                            )
                            # The online consistency trigger requires a probe for
                            # every plan, independently of visualization cadence.
                            if save_scheduled_probe or task9_consistency_enabled:
                                probe_dir = output / "world_probe" / f"task{suite_task_id:02d}" / f"trial{trial:02d}"
                            base_prediction_seed = args.seed + trial + step
                            selected_prediction_seed = base_prediction_seed
                            if (
                                world_consistency_retry_pending
                                and args.world_consistency_retry_candidates > 1
                            ):
                                candidate_seeds = [
                                    base_prediction_seed + candidate_index * 1_000_003
                                    for candidate_index in range(
                                        args.world_consistency_retry_candidates
                                    )
                                ]
                                candidate_chunks = [
                                    predict_action_chunk(
                                        policy, history, instruction,
                                        adapter.proprio(obs), context_frames,
                                        context_stride, action_horizon,
                                        args.num_inference_steps, fps, candidate_seed,
                                        world_frame_stride=world_frame_stride,
                                        video_flip=args.video_flip,
                                        force_context_tokens=(
                                            1 if replan_count == 0 else None
                                        ),
                                        executed_actions=executed_actions,
                                        disable_action_history=(
                                            args.disable_action_history
                                        ),
                                        disable_world_future_tokens=(
                                            args.disable_world_future_tokens
                                        ),
                                        cached_text_context=cached_text_context,
                                        cached_text_mask=cached_text_mask,
                                        cached_world_condition=(
                                            cached_world_condition
                                        ),
                                    )
                                    for candidate_seed in candidate_seeds
                                ]
                                selected_candidate, retry_record = (
                                    select_consensus_action_candidate(
                                        candidate_chunks,
                                        lookahead_steps=(
                                            args.world_consistency_retry_lookahead
                                        ),
                                    )
                                )
                                selected_prediction_seed = candidate_seeds[
                                    selected_candidate
                                ]
                                retry_record.update({
                                    "task_id": suite_task_id,
                                    "trial": trial,
                                    "step": step,
                                    "policy_step": policy_step,
                                    "candidate_seeds": candidate_seeds,
                                    "selected_seed": selected_prediction_seed,
                                })
                                world_consistency_retry_records.append(retry_record)
                                LOG.info(
                                    "task=9 trial=%d step=%d retry consensus "
                                    "selected candidate=%d/%d disagreement=%.6f",
                                    trial, step, selected_candidate + 1,
                                    len(candidate_seeds),
                                    retry_record[
                                        "selected_mean_action_disagreement"
                                    ],
                                )
                            world_consistency_retry_pending = False
                            chunk = predict_action_chunk(
                                policy, history, instruction, adapter.proprio(obs),
                                context_frames, context_stride, action_horizon,
                                args.num_inference_steps, fps,
                                selected_prediction_seed,
                                world_frame_stride=world_frame_stride,
                                world_probe_dir=probe_dir,
                                world_probe_meta={
                                    "task_id": suite_task_id,
                                    "trial": trial,
                                    "step": step,
                                    "world_consistency_prediction_phase": (
                                        current_world_consistency_phase
                                    ),
                                },
                                world_probe_pending=pending_world_probes,
                                world_probe_save_visualization=save_scheduled_probe,
                                video_flip=args.video_flip,
                                force_context_tokens=1 if replan_count == 0 else None,
                                executed_actions=executed_actions,
                                disable_action_history=args.disable_action_history,
                                disable_world_future_tokens=args.disable_world_future_tokens,
                                cached_text_context=cached_text_context,
                                cached_text_mask=cached_text_mask,
                                cached_world_condition=cached_world_condition,
                                paper_recorder=paper_recorder,
                            )
                            if args.oracle_adaptive_replan:
                                current_oracle_replan = oracle_replan_decision(
                                    env,
                                    obs,
                                    far_replan_steps=args.oracle_far_replan_steps,
                                    place_replan_steps=args.oracle_place_replan_steps,
                                    place_xy_threshold=args.oracle_place_xy_threshold,
                                    require_active_goal=not args.oracle_no_require_active_goal,
                                    require_grasped=not args.oracle_no_require_grasped,
                                )
                                raw_selected_replan_steps = current_oracle_replan["selected_replan_steps"]
                                raw_place_phase = bool(current_oracle_replan.get("place_phase"))
                                if args.oracle_one_way_switch:
                                    if raw_place_phase and not oracle_long_mode:
                                        oracle_long_mode = True
                                        oracle_switch_step = step
                                    selected_replan_steps = (
                                        args.oracle_place_replan_steps
                                        if oracle_long_mode else args.oracle_far_replan_steps
                                    )
                                else:
                                    selected_replan_steps = raw_selected_replan_steps
                                current_oracle_replan.update({
                                    "task_id": suite_task_id,
                                    "trial": trial,
                                    "step": step,
                                    "replan_index": replan_count,
                                    "raw_selected_replan_steps": raw_selected_replan_steps,
                                    "selected_replan_steps": selected_replan_steps,
                                    "one_way_switch": bool(args.oracle_one_way_switch),
                                    "long_mode": bool(oracle_long_mode),
                                    "switch_step": oracle_switch_step,
                                })
                                oracle_replan_records.append(dict(current_oracle_replan))
                            else:
                                current_oracle_replan = None
                                selected_replan_steps = args.replan_steps
                            replan_count += 1
                            if args.temporal_ensemble:
                                action_chunks.append((policy_step, chunk))
                                next_replan_policy_step = policy_step + selected_replan_steps
                            else:
                                pending = list(chunk[:selected_replan_steps])
                        if args.temporal_ensemble:
                            action_chunks = [
                                (start_step, saved_chunk)
                                for start_step, saved_chunk in action_chunks
                                if policy_step < start_step + len(saved_chunk)
                            ]
                            temporal_output = temporal_ensemble_action(
                                action_chunks,
                                policy_step,
                                args.temporal_ensemble_decay,
                                disagreement_aware=args.temporal_disagreement_aware,
                                disagreement_window=args.temporal_disagreement_window,
                                position_low=args.temporal_position_low,
                                position_high=args.temporal_position_high,
                                rotation_low=args.temporal_rotation_low,
                                rotation_high=args.temporal_rotation_high,
                                direction_cosine_threshold=(
                                    args.temporal_direction_cosine_threshold
                                ),
                                return_diagnostics=args.temporal_disagreement_aware,
                            )
                            if args.temporal_disagreement_aware:
                                action, temporal_record = temporal_output
                                temporal_disagreement_records.append(temporal_record)
                                all_temporal_disagreement_records.append(temporal_record)
                            else:
                                action = temporal_output
                        else:
                            action = pending.pop(0)
                    if args.save_training_rollouts and step >= args.wait_steps:
                        for camera in adapter.cameras:
                            raw_key = CAMERA_KEYS[camera]
                            rollout_camera_frames[camera].append(
                                np.ascontiguousarray(obs[raw_key], dtype=np.uint8)
                            )
                        rollout_robot_states.append(adapter.proprio(obs).copy())
                        rollout_actions.append(
                            np.asarray(action, dtype=np.float32).copy()
                        )
                    if paper_recorder is not None and step >= args.wait_steps:
                        paper_recorder.observe(step, frame, adapter.proprio(obs), action,
                            collect_success_diagnostics(env, obs, step=step, action=action, done=False))
                    obs, _, done, _ = env.step(action.tolist())
                    online_consistency_events: list[dict[str, Any]] = []
                    if pending_world_probes:
                        online_consistency_events = update_pending_world_probes(
                            pending_world_probes,
                            completed_world_probe_records,
                            policy,
                            observed_step=step + 1,
                            frame=adapter.frame(obs),
                            online_consistency_horizon=(
                                args.world_consistency_horizon
                                if task9_consistency_enabled else None
                            ),
                        )
                    if task9_consistency_enabled:
                        phase, world_consistency_disabled_after_grasp = (
                            task9_world_consistency_phase(
                                env,
                                obs,
                                far_distance=args.world_consistency_far_distance,
                                disabled_after_grasp=(
                                    world_consistency_disabled_after_grasp
                                ),
                            )
                        )
                        current_world_consistency_phase = phase
                        for online_event in online_consistency_events:
                            event = dict(online_event)
                            score = event.get("pred_gt_token_cosine")
                            observed_policy_step = max(
                                0, int(event["observed_step"]) - args.wait_steps
                            )
                            event.update(phase)
                            event.update(world_consistency_replan_decision(
                                event,
                                phase,
                                policy_step=observed_policy_step,
                                soft_threshold=(
                                    args.world_consistency_token_cosine_threshold
                                ),
                                hard_threshold=(
                                    args.world_consistency_hard_token_cosine_threshold
                                ),
                                min_progress=args.world_consistency_min_progress,
                                cooldown_steps=(
                                    args.world_consistency_cooldown_steps
                                ),
                                max_triggers=args.world_consistency_max_triggers,
                                max_policy_step=(
                                    args.world_consistency_max_policy_step
                                ),
                                last_trigger_step=(
                                    world_consistency_last_trigger_step
                                ),
                                trigger_count=world_consistency_trigger_count,
                                episode_done=bool(done),
                            ))
                            world_consistency_records.append(event)
                            if event["triggered"]:
                                world_consistency_trigger_count += 1
                                world_consistency_last_trigger_step = (
                                    observed_policy_step
                                )
                                world_consistency_retry_pending = True
                                # Discard only unexecuted commands. The next
                                # control iteration will immediately generate a
                                # fresh DeltaWorld future and action chunk.
                                pending.clear()
                                action_chunks.clear()
                                LOG.info(
                                    "task=9 trial=%d step=%d early replan: "
                                    "f%d cosine=%.6f, reason=%s, progress=%s, "
                                    "eef-distance=%.4fm, trigger=%d/%d",
                                    trial, step + 1,
                                    args.world_consistency_horizon, float(score),
                                    event["decision_reason"],
                                    (
                                        f'{event["goal_bowl_distance_progress"]:.4f}m'
                                        if event["goal_bowl_distance_progress"] is not None
                                        else "unavailable"
                                    ),
                                    phase["goal_bowl_eef_distance"],
                                    world_consistency_trigger_count,
                                    args.world_consistency_max_triggers,
                                )
                    # Wait/dummy controls are intentionally excluded: the history
                    # condition begins only after the policy has executed a command.
                    if step >= args.wait_steps and action_history_steps > 0:
                        executed_actions.append(np.asarray(action, dtype=np.float32).copy())
                    if diagnostic_stream is not None:
                        diagnostics = collect_success_diagnostics(
                            env,
                            obs,
                            step=step,
                            action=action,
                            done=bool(done),
                            initial_positions=diagnostic_initial_positions,
                        )
                        diagnostics["phase"] = "wait" if step < args.wait_steps else "policy"
                        if current_oracle_replan is not None:
                            diagnostics["oracle_replan"] = current_oracle_replan
                        diagnostic_records.append(diagnostics)
                        diagnostic_stream.write(json.dumps(diagnostics) + "\n")
                        diagnostic_stream.flush()
                    if done:
                        success = True
                        break
                # A probe close to an episode boundary may have only a prefix
                # of its horizon available; retain those aligned comparisons.
                if paper_recorder is not None:
                    paper_recorder.observe(step + 1, adapter.frame(obs), adapter.proprio(obs))
                    paper_recorder.close(success)
                for probe in pending_world_probes:
                    completed_world_probe_records.append(
                        finalize_world_probe_gt(probe, policy)
                    )
                pending_world_probes.clear()
                if diagnostic_stream is not None:
                    diagnostic_stream.close()
                    diagnostic_summary = summarize_success_diagnostics(diagnostic_records)
                    diagnostic_summary_path.write_text(
                        json.dumps(diagnostic_summary, indent=2), encoding="utf-8"
                    )
                else:
                    diagnostic_summary = None
                episode_record = {"trial": trial, "success": success}
                if args.save_world_probe or task9_consistency_enabled:
                    episode_record["world_probe_gt"] = summarize_world_probe_gt(
                        completed_world_probe_records
                    )
                if task9_consistency_enabled:
                    episode_record["task9_world_consistency_replan"] = {
                        "checks": len(world_consistency_records),
                        "triggers": sum(
                            bool(record.get("triggered"))
                            for record in world_consistency_records
                        ),
                        "disabled_after_grasp": bool(
                            world_consistency_disabled_after_grasp
                        ),
                        "trigger_budget_exhausted": bool(
                            world_consistency_trigger_count
                            >= args.world_consistency_max_triggers
                        ),
                        "retry_selections": world_consistency_retry_records,
                        "records": world_consistency_records,
                    }
                if args.temporal_disagreement_aware:
                    episode_record["temporal_disagreement"] = (
                        summarize_temporal_disagreement(temporal_disagreement_records)
                    )
                if args.oracle_adaptive_replan:
                    oracle_summary = {
                        "num_replans": len(oracle_replan_records),
                        "selected_replan_counts": {},
                        "place_phase_replans": sum(bool(record.get("place_phase")) for record in oracle_replan_records),
                        "one_way_switch": bool(args.oracle_one_way_switch),
                        "switched_to_long": any(bool(record.get("long_mode")) for record in oracle_replan_records),
                        "switch_step": oracle_switch_step,
                        "records": oracle_replan_records,
                    }
                    for record in oracle_replan_records:
                        key = str(record.get("selected_replan_steps"))
                        oracle_summary["selected_replan_counts"][key] = oracle_summary["selected_replan_counts"].get(key, 0) + 1
                    episode_record["oracle_adaptive_replan"] = oracle_summary
                if diagnostic_path is not None and diagnostic_summary_path is not None:
                    episode_record.update({
                        "success_diagnostics": str(diagnostic_path),
                        "success_diagnostics_summary": str(diagnostic_summary_path),
                        "success_diagnostics_brief": diagnostic_summary,
                    })
                records.append(episode_record)
                total_success += int(success)
                if args.save_training_rollouts:
                    task_name = Path(task.bddl_file).stem
                    rollout_path = (
                        output
                        / "training_rollouts"
                        / args.suite
                        / f"{task_name}_demo.hdf5"
                    )
                    saved_rollout = save_training_rollout(
                        rollout_path,
                        camera_frames=rollout_camera_frames,
                        robot_states=rollout_robot_states,
                        actions=rollout_actions,
                        instruction=task_instruction,
                        task_name=task_name,
                        task_id=suite_task_id,
                        trial=trial,
                        success=success,
                        checkpoint=str(args.checkpoint),
                    )
                    episode_record["training_rollout"] = (
                        str(saved_rollout) if saved_rollout is not None else None
                    )
                total_trials += 1
                LOG.info("task=%d trial=%d success=%s", suite_task_id, trial, success)
                if args.save_video:
                    save_video(output / "videos" / f"task{suite_task_id:02d}_trial{trial:02d}.mp4", rendered, fps)
                task_result = {
                    "description": task_instruction,
                    "task_instruction": task_instruction,
                    "bddl_instruction": bddl_instruction,
                    "instruction_source": args.instruction_source,
                    "instruction": instruction,
                    "instruction_overridden": args.instruction_override is not None,
                    "success_rate": sum(record["success"] for record in records) / max(len(records), 1),
                    "episodes": records,
                }
                if suite_task_id in task_metadata:
                    task_result["libero_plus_classification"] = task_metadata[suite_task_id]
                results["tasks"][str(suite_task_id)] = task_result
                update_partial_results(results, output)
        finally:
            env.close()
        results["tasks"][str(suite_task_id)] = {
            "description": task_instruction,
            "task_instruction": task_instruction,
            "bddl_instruction": bddl_instruction,
            "instruction_source": args.instruction_source,
            "instruction": instruction,
            "instruction_overridden": args.instruction_override is not None,
            "success_rate": sum(record["success"] for record in records) / max(len(records), 1),
            "episodes": records,
        }
        if suite_task_id in task_metadata:
            results["tasks"][str(suite_task_id)]["libero_plus_classification"] = (
                task_metadata[suite_task_id]
            )
    total_success = sum(
        int(bool(episode.get("success")))
        for task_result in results["tasks"].values()
        for episode in task_result.get("episodes", [])
    )
    total_trials = sum(
        len(task_result.get("episodes", []))
        for task_result in results["tasks"].values()
    )
    results.update(success_rate=total_success / max(total_trials, 1),
                   successes=total_success, trials=total_trials)
    libero_plus_summary = summarize_classified_results(results["tasks"])
    results["libero_plus_summary"] = libero_plus_summary
    temporal_disagreement_summary = (
        summarize_temporal_disagreement(all_temporal_disagreement_records)
        if args.temporal_disagreement_aware else None
    )
    results["temporal_disagreement_summary"] = temporal_disagreement_summary
    stats = {
        "checkpoint": str(args.checkpoint),
        "action_checkpoint_world": policy.action_checkpoint_world,
        "effective_world_checkpoint": policy.eval_world_checkpoint,
        "world_checkpoint_overridden": bool(
            policy.world_checkpoint_overridden
        ),
        "config": str(args.config),
        "suite": args.suite,
        "benchmark_family": args.benchmark_family,
        "benchmark_base_suite": args.benchmark_base_suite,
        "benchmark_perturbation": args.benchmark_perturbation,
        "instruction_source": args.instruction_source,
        "task_indices": list(task_indices),
        "task_selection": task_selection,
        "random_task": bool(args.random_task),
        "random_task_seed": args.random_task_seed if args.random_task_seed is not None else args.seed,
        "world_prediction_steps": int(policy.network.world_prediction_steps),
        "world_frame_stride": int(world_frame_stride),
        "action_history_steps": action_history_steps,
        "action_history_source": "actually_executed_policy_commands",
        "action_history_disabled": bool(args.disable_action_history),
        "world_future_tokens_disabled": bool(args.disable_world_future_tokens),
        "effective_world_future_tokens": (
            0 if args.disable_world_future_tokens else int(policy.network.world_prediction_steps)
        ),
        "warm_start_available_context": True,
        "instruction_override": args.instruction_override,
        "run_name": run_name,
        "seed": int(args.seed),
        "success_definition": "success_once_env_done",
        "output_dir": str(output),
        "success_rate": results["success_rate"],
        "successes": total_success,
        "trials": total_trials,
        "num_tasks": len(results["tasks"]),
        "num_trials_per_task_requested": args.num_trials,
        "replan_steps": args.replan_steps,
        "temporal_ensemble": bool(args.temporal_ensemble),
        "temporal_ensemble_decay": float(args.temporal_ensemble_decay),
        "temporal_disagreement_aware": bool(args.temporal_disagreement_aware),
        "temporal_disagreement_window": int(args.temporal_disagreement_window),
        "temporal_position_thresholds": [args.temporal_position_low, args.temporal_position_high],
        "temporal_rotation_thresholds": [args.temporal_rotation_low, args.temporal_rotation_high],
        "temporal_direction_cosine_threshold": float(
            args.temporal_direction_cosine_threshold
        ),
        "temporal_disagreement_summary": temporal_disagreement_summary,
        "libero_plus_summary": libero_plus_summary,
        "num_inference_steps": args.num_inference_steps,
        "max_steps": args.max_steps,
        "wait_steps": args.wait_steps,
        "video_flip": args.video_flip,
        "save_video": bool(args.save_video),
        "save_world_probe": bool(args.save_world_probe),
        "effective_world_probe": bool(
            args.save_world_probe or args.task9_world_consistency_replan
        ),
        "task9_world_consistency_replan": bool(
            args.task9_world_consistency_replan
        ),
        "world_consistency_horizon": int(args.world_consistency_horizon),
        "world_consistency_token_cosine_threshold": float(
            args.world_consistency_token_cosine_threshold
        ),
        "world_consistency_hard_token_cosine_threshold": float(
            args.world_consistency_hard_token_cosine_threshold
        ),
        "world_consistency_far_distance": float(
            args.world_consistency_far_distance
        ),
        "world_consistency_min_progress": float(
            args.world_consistency_min_progress
        ),
        "world_consistency_cooldown_steps": int(
            args.world_consistency_cooldown_steps
        ),
        "world_consistency_max_triggers": int(
            args.world_consistency_max_triggers
        ),
        "world_consistency_max_policy_step": int(
            args.world_consistency_max_policy_step
        ),
        "world_consistency_retry_candidates": int(
            args.world_consistency_retry_candidates
        ),
        "world_consistency_retry_lookahead": int(
            args.world_consistency_retry_lookahead
        ),
        "save_success_diagnostics": bool(args.save_success_diagnostics),
        "oracle_adaptive_replan": bool(args.oracle_adaptive_replan),
        "oracle_one_way_switch": bool(args.oracle_one_way_switch),
        "oracle_far_replan_steps": int(args.oracle_far_replan_steps),
        "oracle_place_replan_steps": int(args.oracle_place_replan_steps),
        "oracle_place_xy_threshold": float(args.oracle_place_xy_threshold),
        "oracle_require_active_goal": not args.oracle_no_require_active_goal,
        "oracle_require_grasped": not args.oracle_no_require_grasped,
        "world_probe_every_replan": int(args.world_probe_every_replan),
        "tasks": {
            task_id: {
                "description": task_result["description"],
                "task_instruction": task_result["task_instruction"],
                "bddl_instruction": task_result["bddl_instruction"],
                "instruction_source": task_result["instruction_source"],
                "instruction": task_result["instruction"],
                "instruction_overridden": task_result["instruction_overridden"],
                "success_rate": task_result["success_rate"],
                "successes": sum(int(ep["success"]) for ep in task_result["episodes"]),
                "trials": len(task_result["episodes"]),
                **(
                    {
                        "world_consistency_checks": sum(
                            int(ep["task9_world_consistency_replan"]["checks"])
                            for ep in task_result["episodes"]
                            if "task9_world_consistency_replan" in ep
                        ),
                        "world_consistency_triggers": sum(
                            int(ep["task9_world_consistency_replan"]["triggers"])
                            for ep in task_result["episodes"]
                            if "task9_world_consistency_replan" in ep
                        ),
                    }
                    if any(
                        "task9_world_consistency_replan" in ep
                        for ep in task_result["episodes"]
                    ) else {}
                ),
                **(
                    {"libero_plus_classification": task_result["libero_plus_classification"]}
                    if "libero_plus_classification" in task_result else {}
                ),
            }
            for task_id, task_result in results["tasks"].items()
        },
    }
    results["progress"] = {
        "status": "complete",
        "completed_trials": total_trials,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(output / "results.json", results)
    atomic_write_json(output / "rollout_stats.json", stats)
    update_partial_results(results, output, status="complete")
    LOG.info("success_rate=%.4f (%d/%d)", results["success_rate"], total_success, total_trials)
    LOG.info("wrote rollout stats to %s", output / "rollout_stats.json")


if __name__ == "__main__":
    main()
