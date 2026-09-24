#!/usr/bin/env python3
"""Black-box system-efficiency benchmark for robot policies.

The primary unit under test is one complete tensor-to-tensor model inference:
prepared tensors already resident on the device to an action chunk left on the
device. The built-in adapter uses the exact DeltaWorld-conditioned ActionDiT
inference path used by ``eval_feature_action_libero.py``. Other architectures
can be compared under the same harness with ``--adapter-factory module:function``.

This script deliberately does not report component-level timings as headline
metrics. Synchronized wall time is the main model-only latency; CUDA event time
is retained as a device-compute diagnostic. Observation preprocessing, text
tokenization, host/device transfer, simulator work, and action transfer back to
the host are excluded.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import importlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("policy_efficiency")


@dataclass
class PolicyResult:
    """Normalized output of one policy query.

    Streaming/autoregressive adapters may provide ``ttfa_ms`` measured at the
    instant their first executable action becomes available.  Atomic chunk
    policies should leave it unset; the harness then sets TTFA equal to TTFC.
    """

    actions: Any
    ttfa_ms: float | None = None
    events: dict[str, Any] | None = None
    # Internal CUDA markers resolved by the harness after its normal terminal
    # synchronization. They are never serialized directly.
    component_cuda_events: dict[str, tuple[Any, Any]] | None = None


@runtime_checkable
class PolicyBenchmarkAdapter(Protocol):
    """Minimal black-box interface used for cross-architecture comparisons."""

    model: Any
    generates_full_chunk_atomically: bool

    def reset(self, phase: str) -> None:
        """Prepare an episode-context cold or steady-state query."""

    def predict_action(self) -> PolicyResult | np.ndarray | dict[str, Any]:
        """Consume prepared device tensors and return a device action chunk."""

    def metadata(self) -> dict[str, Any]:
        """Return architecture and input/output metadata."""


class FeatureActionAdapter:
    """Black-box adapter for the deployed DeltaWorld + ActionDiT policy."""

    generates_full_chunk_atomically = True

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        # Keep LIBERO / Lightning dependencies out of external-adapter runs.
        from scripts.eval_feature_action_libero import (
            CAMERA_KEYS,
            LiberoObservationAdapter,
            dataset_args,
            load_policy,
        )

        self._dataset_args = dataset_args
        self._camera_keys = CAMERA_KEYS
        self.device = device
        self.config_path = Path(args.config)
        self.checkpoint_path = Path(args.checkpoint)
        self.policy, self.config = load_policy(
            args.config,
            args.checkpoint,
            device,
            world_prediction_steps_override=args.world_prediction_steps_override,
        )
        self.model = self.policy
        data = self._dataset_args(self.config)
        frame_size = self.config["data"]["init_args"].get("frame_size", 512)
        self.observation_adapter = LiberoObservationAdapter(
            data.get("cameras", ["agentview_rgb"]),
            data.get("camera_layout", "first"),
            frame_size,
        )
        self.context_frames = int(data.get("context_frames", 4))
        self.context_stride = int(data.get("context_frame_stride", 2))
        self.world_frame_stride = int(
            data.get("world_frame_stride", self.context_stride)
        )
        self.action_horizon = int(data.get("action_horizon", self.policy.action_horizon))
        self.fps = float(data.get("fps", args.fps))
        self.history_len = (self.context_frames - 1) * self.context_stride + 1
        self.action_history_steps = int(self.policy.network.action_history_steps)
        self.inference_steps = int(args.num_inference_steps)
        self.instruction = str(args.instruction)
        self.seed = int(args.seed)
        self.profile_components = bool(args.profile_components)
        self.call_index = 0
        self._phase = "steady"
        self._prepared: dict[str, Any] = {}
        self._raw_observations = self._make_raw_observations(
            count=max(self.history_len + 2, 10), seed=self.seed
        )
        (
            self._cached_action_text_context,
            self._cached_action_text_mask,
            self._cached_world_condition,
        ) = self._cache_static_text_conditions()

    def _make_raw_observations(self, count: int, seed: int) -> list[dict[str, np.ndarray]]:
        rng = np.random.default_rng(seed)
        observations = []
        for _ in range(count):
            observation = {
                key: rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
                for key in self._camera_keys.values()
            }
            observation.update({
                "robot0_gripper_qpos": rng.normal(size=2).astype(np.float32),
                "robot0_eef_pos": rng.normal(size=3).astype(np.float32),
                "robot0_eef_quat": rng.normal(size=4).astype(np.float32),
            })
            observations.append(observation)
        return observations

    def reset(self, phase: str) -> None:
        """Prepare all non-model work outside the timed region."""
        if phase not in {"cold", "steady"}:
            raise ValueError(f"unknown benchmark phase: {phase}")
        self._phase = phase
        raw_count = 1 if phase == "cold" else self.history_len
        frames = [
            self.observation_adapter.frame(observation)
            for observation in self._raw_observations[:raw_count]
        ]
        available_context = min(
            self.context_frames, 1 + (len(frames) - 1) // self.context_stride
        )
        indices = [
            len(frames) - 1 - (available_context - 1 - index) * self.context_stride
            for index in range(available_context)
        ]
        stacked = np.stack([frames[index] for index in indices])
        if stacked.ndim == 4:
            observed = torch.from_numpy(stacked).permute(0, 3, 1, 2)[None]
        elif stacked.ndim == 5:
            observed = torch.from_numpy(stacked).permute(0, 1, 4, 2, 3)[None]
        else:
            raise ValueError(f"unexpected prepared observation shape: {stacked.shape}")
        observed = observed.to(self.device)
        observed_times = torch.tensor(
            indices, dtype=torch.float32, device=self.device
        )[None] / self.fps
        future_offsets = torch.arange(
            1,
            int(self.policy.network.world_prediction_steps) + 1,
            dtype=torch.float32,
            device=self.device,
        )[None]
        future_times = observed_times[:, -1:] + future_offsets * (
            self.world_frame_stride / self.fps
        )
        raw_observation = self._raw_observations[raw_count - 1]
        proprio = torch.from_numpy(
            self.observation_adapter.proprio(raw_observation)
        ).unsqueeze(0).to(self.device)
        action_history = action_history_mask = None
        if self.action_history_steps:
            action_history = torch.zeros(
                1,
                self.action_history_steps,
                self.policy.network.action_dit.action_dim,
                device=self.device,
            )
            action_history_mask = torch.ones(
                1, self.action_history_steps, dtype=torch.bool, device=self.device
            )
        action_dtype = self.policy.network.action_dit.action_encoder.weight.dtype
        initial_actions = torch.randn(
            1,
            self.action_horizon,
            self.policy.network.action_dit.action_dim,
            device=self.device,
            dtype=action_dtype,
        )
        timesteps, deltas = self.policy.network.action_scheduler.build_inference_schedule(
            self.inference_steps, self.device, action_dtype
        )
        self._prepared = {
            "observed": observed,
            "observed_times": observed_times,
            "future_times": future_times,
            "proprio": proprio,
            "action_history": action_history,
            "action_history_mask": action_history_mask,
            "initial_actions": initial_actions,
            "timesteps": timesteps,
            "deltas": deltas,
            "generator": torch.Generator(device=self.device).manual_seed(
                self.seed + self.call_index
            ),
        }

    def _tokenize(self, encoder) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = encoder.tokenizer(
            [self.instruction],
            padding=True,
            truncation=True,
            max_length=encoder.max_length,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        )

    @staticmethod
    def _encode_prepared_text(encoder, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = inputs
        hidden = encoder.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        projection = encoder.projection
        hidden = hidden.to(device=projection.weight.device, dtype=projection.weight.dtype)
        tokens = projection(hidden)
        return tokens, attention_mask.to(device=tokens.device, dtype=torch.bool)

    def _cache_static_text_conditions(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run episode-static T5 encoders once, outside per-query timing."""
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            action_text_context, action_text_mask = self._encode_prepared_text(
                self.policy.instruction_encoder,
                self._tokenize(self.policy.instruction_encoder),
            )
            world_condition = None
            world_encoder = getattr(
                self.policy.network.world_rollout.delta_world,
                "task_film_encoder",
                None,
            )
            if world_encoder is not None:
                world_condition, _ = self._encode_prepared_text(
                    world_encoder, self._tokenize(world_encoder)
                )
        return action_text_context, action_text_mask, world_condition

    def _action_native_context(
        self, prepared: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ActionDiT-native text/proprio tokens from cached text."""
        text_context = self._cached_action_text_context
        text_mask = self._cached_action_text_mask
        if self.policy.proprio_encoder is not None:
            proprio_token = self.policy.proprio_encoder(
                prepared["proprio"]
            ).unsqueeze(1)
            text_context = torch.cat([text_context, proprio_token], dim=1)
            text_mask = torch.cat([
                text_mask,
                torch.ones(
                    proprio_token.shape[:2], dtype=torch.bool, device=self.device
                ),
            ], dim=1)
        return text_context, text_mask

    def _sample_actions(self, condition, prepared: dict[str, Any]) -> torch.Tensor:
        """Run the complete iterative ActionDiT flow-matching sampler."""
        actions = prepared["initial_actions"].clone()
        for timestep, delta in zip(prepared["timesteps"], prepared["deltas"]):
            velocity = self.policy.network(
                actions, timestep.expand(actions.shape[0]), condition
            )
            actions = self.policy.network.action_scheduler.step(
                velocity, delta, actions
            )
        return actions

    def predict_action(self) -> PolicyResult:
        """Run neural/sampler computation on device-resident tensors.

        When component profiling is enabled, CUDA markers divide the unchanged
        inference path at the output of DeltaWorld. Thus ActionDiT time includes
        native text/proprio conditioning, external K/V cache construction, and
        every flow-matching denoising step.
        """
        prepared = self._prepared
        component_events = None
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            world_start = world_end = action_end = None
            if self.profile_components:
                world_start = torch.cuda.Event(enable_timing=True)
                world_end = torch.cuda.Event(enable_timing=True)
                action_end = torch.cuda.Event(enable_timing=True)
                world_start.record()

            world_output = self.policy.network.rollout_world(
                prepared["observed"],
                observed_timestamps=prepared["observed_times"],
                future_timestamps=prepared["future_times"],
                generator=prepared["generator"],
                instruction=self._cached_world_condition,
            )
            if world_end is not None:
                world_end.record()

            text_context, text_mask = self._action_native_context(prepared)
            condition = self.policy.network.build_action_condition(
                world_output,
                text_context=text_context,
                text_mask=text_mask,
                action_history=prepared["action_history"],
                action_history_mask=prepared["action_history_mask"],
            )
            actions = self._sample_actions(condition, prepared)
            if action_end is not None:
                action_end.record()
                component_events = {
                    "deltaworld": (world_start, world_end),
                    "action_dit": (world_end, action_end),
                }
        self.call_index += 1
        return PolicyResult(
            actions=actions[0], component_cuda_events=component_events
        )

    def metadata(self) -> dict[str, Any]:
        data = self._dataset_args(self.config)
        return {
            "adapter": "feature_action",
            "architecture": "DeltaWorld-conditioned ActionDiT",
            "config": str(self.config_path),
            "checkpoint": str(self.checkpoint_path),
            "cameras": list(data.get("cameras", ["agentview_rgb"])),
            "camera_layout": data.get("camera_layout", "first"),
            "frame_size": self.config["data"]["init_args"].get("frame_size", 512),
            "context_frames": self.context_frames,
            "context_frame_stride": self.context_stride,
            "world_prediction_steps": int(self.policy.network.world_prediction_steps),
            "world_frame_stride": self.world_frame_stride,
            "action_horizon": self.action_horizon,
            "num_inference_steps": self.inference_steps,
            "fps": self.fps,
            "generates_full_chunk_atomically": True,
            "latency_scope": "model_only_device_tensor_to_device_tensor",
            "text_tokenization_included": False,
            "t5_encoder_forward_included": False,
            "text_embedding_cached": True,
            "component_latency_boundary": {
                "deltaworld": "VFM/DeltaTok encoding plus DeltaWorld rollout",
                "action_dit": (
                    "text/proprio native conditioning plus external K/V cache "
                    "construction plus iterative ActionDiT sampling"
                ),
            },
        }


def import_factory(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("--adapter-factory must have form module:function")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"adapter factory is not callable: {spec}")
    return factory


def normalize_result(value: Any) -> PolicyResult:
    if isinstance(value, PolicyResult):
        result = value
    elif isinstance(value, dict):
        if "actions" not in value:
            raise KeyError("policy result dict must contain 'actions'")
        result = PolicyResult(
            actions=value["actions"],
            ttfa_ms=(None if value.get("ttfa_ms") is None else float(value["ttfa_ms"])),
            events=(None if value.get("events") is None else dict(value["events"])),
        )
    else:
        result = PolicyResult(actions=value)
    actions = result.actions
    if not torch.is_tensor(actions):
        actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[0] <= 0:
        raise ValueError(f"policy must return [action_horizon, action_dim], got {actions.shape}")
    result.actions = actions
    return result


def begin_phase(adapter: PolicyBenchmarkAdapter, phase: str, count: int) -> None:
    """Start a phase for adapters with persistent inference state."""
    method = getattr(adapter, "begin_phase", None)
    if callable(method):
        method(phase, count)


def prepare_query(adapter: PolicyBenchmarkAdapter, phase: str, index: int) -> None:
    """Prepare non-model inputs without necessarily resetting model state."""
    method = getattr(adapter, "prepare_query", None)
    if callable(method):
        method(phase, index)
    else:
        adapter.reset(phase)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None,
                "p50": None, "p90": None, "p99": None,
                "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def timed_query(
    adapter: PolicyBenchmarkAdapter,
    device: torch.device,
    phase: str,
    index: int,
) -> tuple[dict[str, Any], PolicyResult]:
    prepare_query(adapter, phase, index)
    synchronize(device)
    start_event = end_event = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    wall_start = time.perf_counter_ns()
    result = normalize_result(adapter.predict_action())
    if end_event is not None:
        end_event.record()
    synchronize(device)
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
    gpu_ms = float(start_event.elapsed_time(end_event)) if start_event is not None else None
    ttfa_ms = result.ttfa_ms
    if ttfa_ms is None and bool(adapter.generates_full_chunk_atomically):
        ttfa_ms = wall_ms
    record = {
        "phase": phase,
        "wall_ms": wall_ms,
        "gpu_ms": gpu_ms,
        "ttfa_ms": ttfa_ms,
        "ttfc_ms": wall_ms,
        "returned_action_steps": int(result.actions.shape[0]),
        "action_dim": int(result.actions.shape[1]),
    }
    if result.events:
        record["events"] = result.events
    if result.component_cuda_events:
        component_gpu_ms = {
            name: float(start.elapsed_time(end))
            for name, (start, end) in result.component_cuda_events.items()
        }
        attributed_gpu_ms = float(sum(component_gpu_ms.values()))
        record["component_gpu_ms"] = component_gpu_ms
        record["component_attributed_gpu_ms"] = attributed_gpu_ms
        record["component_share_percent"] = {
            name: 100.0 * value / attributed_gpu_ms
            for name, value in component_gpu_ms.items()
        }
        record["component_unattributed_gpu_ms"] = (
            None if gpu_ms is None else float(gpu_ms - attributed_gpu_ms)
        )
        record["wall_minus_attributed_gpu_ms"] = float(wall_ms - attributed_gpu_ms)
    return record, result


def parameter_summary(model: Any) -> dict[str, int | float | None]:
    if not hasattr(model, "parameters"):
        return {
            "total_parameters": None,
            "trainable_parameters": None,
            "parameter_storage_bytes": None,
        }
    parameters = list(model.parameters())
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        "parameter_storage_bytes": int(
            sum(parameter.numel() * parameter.element_size() for parameter in parameters)
        ),
    }


def profile_flops(adapter: PolicyBenchmarkAdapter, device: torch.device) -> float | None:
    """Best-effort profiler FLOPs for one complete steady-state query.

    PyTorch only assigns FLOP formulas to supported operators; fused attention
    and custom kernels can be under-counted.  Results are therefore explicitly
    labelled ``torch_profiler_supported_ops`` in the output and should only be
    compared when every architecture uses this identical method.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    schedule_queries = max(
        1, int(adapter.metadata().get("chunks_per_video_prefill", 1))
    )
    begin_phase(adapter, "steady", schedule_queries)
    synchronize(device)
    with torch.profiler.profile(activities=activities, with_flops=True) as profile:
        for index in range(schedule_queries):
            prepare_query(adapter, "steady", index)
            normalize_result(adapter.predict_action())
        synchronize(device)
    total = float(sum(float(event.flops or 0) for event in profile.key_averages()))
    average = total / schedule_queries
    return average if average > 0 else None


def measure_peak_memory(
    adapter: PolicyBenchmarkAdapter, device: torch.device
) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    schedule_queries = max(
        1, int(adapter.metadata().get("chunks_per_video_prefill", 1))
    )
    begin_phase(adapter, "steady", schedule_queries)
    synchronize(device)
    for index in range(schedule_queries):
        prepare_query(adapter, "steady", index)
        normalize_result(adapter.predict_action())
    synchronize(device)
    return {
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def aggregate_phase(
    records: list[dict[str, Any]], replan_steps: int, fps: float
) -> dict[str, Any]:
    wall = [float(record["wall_ms"]) for record in records]
    gpu = [float(record["gpu_ms"]) for record in records if record["gpu_ms"] is not None]
    ttfa = [float(record["ttfa_ms"]) for record in records if record["ttfa_ms"] is not None]
    budget_ms = 1000.0 * replan_steps / fps
    wall_stats = distribution(wall)
    event_records = [record for record in records if record.get("events")]
    paths = sorted({
        str(record["events"].get("model_path"))
        for record in event_records
        if record["events"].get("model_path") is not None
    })
    path_latency = {
        path: distribution([
            float(record["wall_ms"]) for record in event_records
            if str(record["events"].get("model_path")) == path
        ])
        for path in paths
    }
    return {
        "policy_query_wall_ms": wall_stats,
        "policy_query_gpu_event_ms": distribution(gpu),
        "ttfa_ms": distribution(ttfa),
        "ttfc_ms": wall_stats,
        "amortized_wall_ms_per_executed_action": {
            key: (value / replan_steps if isinstance(value, float) else value)
            for key, value in wall_stats.items()
        },
        "replan_compute_budget_ms": budget_ms,
        "budget_utilization_percent": distribution(
            [100.0 * value / budget_ms for value in wall]
        ),
        "deadline_miss_rate": float(sum(value > budget_ms for value in wall) / len(wall)),
        "policy_query_hz_from_mean": 1000.0 / wall_stats["mean"],
        "compute_supported_action_hz_from_mean": (
            1000.0 * replan_steps / wall_stats["mean"]
        ),
        "schedule": {
            "planner_refresh_rate": (
                float(sum(bool(record["events"].get("planner_refresh"))
                          for record in event_records) / len(event_records))
                if event_records else None
            ),
            "path_query_wall_ms": path_latency,
        },
    }


def aggregate_component_latency(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregate the optional CUDA-event decomposition for one phase."""
    profiled = [record for record in records if record.get("component_gpu_ms")]
    if not profiled:
        return None
    names = ("deltaworld", "action_dit")
    return {
        "clock": "torch_cuda_event",
        "boundary": {
            "deltaworld": "VFM/DeltaTok encoding plus DeltaWorld rollout",
            "action_dit": (
                "text/proprio native conditioning plus external K/V cache "
                "construction plus iterative ActionDiT sampling"
            ),
        },
        "gpu_ms": {
            name: distribution([
                float(record["component_gpu_ms"][name]) for record in profiled
            ])
            for name in names
        },
        "share_of_attributed_gpu_time_percent": {
            name: distribution([
                float(record["component_share_percent"][name]) for record in profiled
            ])
            for name in names
        },
        "attributed_gpu_ms": distribution([
            float(record["component_attributed_gpu_ms"]) for record in profiled
        ]),
        "unattributed_outer_gpu_event_ms": distribution([
            float(record["component_unattributed_gpu_ms"])
            for record in profiled
            if record["component_unattributed_gpu_ms"] is not None
        ]),
        "wall_minus_attributed_gpu_ms": distribution([
            float(record["wall_minus_attributed_gpu_ms"]) for record in profiled
        ]),
        "interpretation": (
            "Component shares use the sum of the two component CUDA-event times. "
            "Python dispatch and terminal synchronization remain visible only in "
            "the complete synchronized wall-clock latency."
        ),
    }


def flatten_csv(summary: dict[str, Any]) -> dict[str, Any]:
    steady = summary["latency"]["steady"]
    cold = summary["latency"]["cold"]
    memory = summary["memory"]
    compute = summary["compute"]
    return {
        "model_name": summary["model_name"],
        "device": summary["runtime"]["device_name"],
        "precision": summary["precision"],
        "replan_steps": summary["replan_steps"],
        "action_horizon": summary["model_metadata"].get("action_horizon"),
        "steady_query_p50_ms": steady["policy_query_wall_ms"]["p50"],
        "steady_query_p99_ms": steady["policy_query_wall_ms"]["p99"],
        "steady_amortized_p50_ms_per_action": (
            steady["amortized_wall_ms_per_executed_action"]["p50"]
        ),
        "steady_budget_utilization_p50_percent": (
            steady["budget_utilization_percent"]["p50"]
        ),
        "steady_deadline_miss_rate": steady["deadline_miss_rate"],
        "cold_query_p50_ms": cold["policy_query_wall_ms"]["p50"],
        "peak_cuda_allocated_gib": (
            memory["peak_cuda_allocated_bytes"] / 2**30
            if memory["peak_cuda_allocated_bytes"] is not None else None
        ),
        "total_parameters": summary["parameters"]["total_parameters"],
        "profiled_gflops_per_query": (
            compute["profiled_flops_per_query"] / 1e9
            if compute["profiled_flops_per_query"] is not None else None
        ),
        "profiled_gflops_per_executed_action": (
            compute["profiled_flops_per_executed_action"] / 1e9
            if compute["profiled_flops_per_executed_action"] is not None else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--feature-action", action="store_true")
    source.add_argument(
        "--adapter-factory",
        help="External black-box adapter factory as module:function; called with (args, device).",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="bf16-eager")
    parser.add_argument("--instruction", default="pick up the black bowl and place it on the plate")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--world-prediction-steps-override", type=int)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--warmup-queries", type=int, default=20)
    parser.add_argument("--steady-queries", type=int, default=100)
    parser.add_argument("--cold-queries", type=int, default=20)
    parser.add_argument("--profile-flops", action="store_true")
    parser.add_argument(
        "--profile-components",
        action="store_true",
        help=(
            "Record CUDA-event DeltaWorld vs ActionDiT timings on the same "
            "queries. Supported by the built-in --feature-action adapter."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--factory-kwargs",
        type=json.loads,
        default={},
        help="JSON object made available to external adapter factories via args.factory_kwargs.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/policy_efficiency"
    )
    args = parser.parse_args()
    if args.feature_action and (args.config is None or args.checkpoint is None):
        parser.error("--feature-action requires --config and --checkpoint")
    if args.profile_components and not args.feature_action:
        parser.error("--profile-components currently requires --feature-action")
    if args.profile_components and not str(args.device).startswith("cuda"):
        parser.error("--profile-components requires a CUDA device")
    if args.replan_steps <= 0 or args.fps <= 0:
        parser.error("--replan-steps and --fps must be positive")
    if min(args.warmup_queries, args.steady_queries, args.cold_queries) < 0:
        parser.error("query counts must be non-negative")
    if args.steady_queries == 0 or args.cold_queries == 0:
        parser.error("--steady-queries and --cold-queries must be positive")
    if not isinstance(args.factory_kwargs, dict):
        parser.error("--factory-kwargs must decode to a JSON object")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if args.feature_action:
        adapter: PolicyBenchmarkAdapter = FeatureActionAdapter(args, device)
    else:
        adapter = import_factory(args.adapter_factory)(args, device)
    for method in ("reset", "predict_action", "metadata"):
        if not callable(getattr(adapter, method, None)):
            raise TypeError(f"benchmark adapter is missing callable {method}()")
    if not hasattr(adapter, "generates_full_chunk_atomically"):
        raise TypeError("benchmark adapter must declare generates_full_chunk_atomically")

    synchronize(device)
    cuda_allocated_after_load = (
        int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None
    )
    LOG.info("warming up complete policy for %d steady-state queries", args.warmup_queries)
    begin_phase(adapter, "steady", args.warmup_queries)
    for index in range(args.warmup_queries):
        prepare_query(adapter, "steady", index)
        normalize_result(adapter.predict_action())
    synchronize(device)

    records: list[dict[str, Any]] = []
    for phase, count in (("cold", args.cold_queries), ("steady", args.steady_queries)):
        LOG.info("measuring %s policy queries: n=%d", phase, count)
        begin_phase(adapter, phase, count)
        for index in range(count):
            record, _ = timed_query(adapter, device, phase, index)
            record["index"] = index
            records.append(record)

    returned_steps = {int(record["returned_action_steps"]) for record in records}
    if len(returned_steps) != 1:
        raise RuntimeError(f"policy returned inconsistent action horizons: {sorted(returned_steps)}")
    returned_horizon = next(iter(returned_steps))
    if args.replan_steps > returned_horizon:
        raise ValueError(
            f"replan_steps={args.replan_steps} exceeds returned action horizon "
            f"{returned_horizon}; amortized latency would be invalid"
        )

    memory = measure_peak_memory(adapter, device)
    memory["cuda_allocated_after_model_load_bytes"] = cuda_allocated_after_load
    flops = profile_flops(adapter, device) if args.profile_flops else None
    metadata = dict(adapter.metadata())
    effective_fps = float(metadata.get("fps", args.fps))
    phase_records = {
        phase: [record for record in records if record["phase"] == phase]
        for phase in ("cold", "steady")
    }
    summary = {
        "schema_version": 1,
        "model_name": args.model_name,
        "precision": args.precision,
        "replan_steps": args.replan_steps,
        "runtime": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "platform": platform.platform(),
        },
        "protocol": {
            "batch_size": 1,
            "warmup_queries": args.warmup_queries,
            "steady_queries": args.steady_queries,
            "cold_queries": args.cold_queries,
            "main_latency_clock": "synchronized_wall_time",
            "gpu_diagnostic_clock": "torch_cuda_event",
            "policy_query_scope": (
                "device-resident prepared tensors through action chunk retained on device"
            ),
            "latency_scope": "model_only",
            "excluded": [
                "observation preprocessing",
                "text tokenization",
                "episode-static T5 encoder forwards and text projections",
                "host-to-device transfer",
                "device-to-host transfer",
                "simulator and robot execution",
            ],
            "included": [
                "cached action-side text tokens and current proprio projection",
                "cached DeltaWorld task condition when configured",
                "VFM and DeltaTok",
                "DeltaWorld rollout and KV construction",
                "ActionDiT iterative sampler",
            ],
            "checkpoint_loading_excluded": True,
            "video_and_diagnostics_excluded": True,
        },
        "model_metadata": metadata,
        "parameters": parameter_summary(adapter.model),
        "memory": memory,
        "latency": {
            phase: aggregate_phase(phase_records[phase], args.replan_steps, effective_fps)
            for phase in ("cold", "steady")
        },
        "component_latency": {
            "enabled": args.profile_components,
            "cold": aggregate_component_latency(phase_records["cold"]),
            "steady": aggregate_component_latency(phase_records["steady"]),
        },
        "compute": {
            "profiled_flops_per_query": flops,
            "profiled_flops_per_executed_action": (
                flops / args.replan_steps if flops is not None else None
            ),
            "flops_method": (
                "torch_profiler_supported_ops" if args.profile_flops else "not_measured"
            ),
            "flops_warning": (
                "Fused attention/custom kernels may be under-counted; compare only identical profiler methods."
                if args.profile_flops else None
            ),
        },
    }

    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in args.model_name)
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    output = args.output_dir / safe_name / run_name
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output / "latency_queries.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    csv_record = flatten_csv(summary)
    with (output / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_record))
        writer.writeheader()
        writer.writerow(csv_record)

    steady = summary["latency"]["steady"]
    LOG.info("wrote benchmark to %s", output)
    LOG.info(
        "steady query p50/p99 %.2f/%.2f ms; amortized p50 %.2f ms/action; "
        "budget p50 %.2f%%; deadline misses %.2f%%",
        steady["policy_query_wall_ms"]["p50"],
        steady["policy_query_wall_ms"]["p99"],
        steady["amortized_wall_ms_per_executed_action"]["p50"],
        steady["budget_utilization_percent"]["p50"],
        100.0 * steady["deadline_miss_rate"],
    )
    if args.profile_components:
        components = summary["component_latency"]["steady"]
        LOG.info(
            "steady component GPU p50: DeltaWorld %.2f ms (%.1f%%), "
            "ActionDiT incl. K/V %.2f ms (%.1f%%)",
            components["gpu_ms"]["deltaworld"]["p50"],
            components["share_of_attributed_gpu_time_percent"]["deltaworld"]["mean"],
            components["gpu_ms"]["action_dit"]["p50"],
            components["share_of_attributed_gpu_time_percent"]["action_dit"]["mean"],
        )


if __name__ == "__main__":
    main()
