#!/usr/bin/env python3
"""Replay LIBERO expert demonstrations and build a simulator-verified whitelist.

The HDF5 ``rewards`` and ``dones`` arrays are deliberately ignored: LIBERO's
dataset conversion writes a positive terminal marker for every converted demo.
This audit restores each demo's XML and initial MuJoCo state, replays its
actions, and asks the environment to recompute task success.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("libero_expert_audit")
DEFAULT_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
)


def configure_libero(home: Path | None) -> None:
    if home is None:
        return
    home = home.expanduser().resolve()
    candidates = (home,) if (home / "libero" / "libero").is_dir() else (
        home,
        home / "libero",
    )
    for candidate in reversed(candidates):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def resolve_assets_root(configured: str, explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    candidates.extend(
        (
            Path(configured).expanduser(),
            Path.home() / ".cache/libero/assets",
        )
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "cannot find LIBERO assets; checked "
        + ", ".join(str(candidate) for candidate in candidates)
        + ". Pass --assets-root explicitly."
    )


def sorted_demo_names(h5: h5py.File) -> list[str]:
    return sorted(
        h5["data"].keys(),
        key=lambda name: (
            (0, int(name.rsplit("_", 1)[-1]))
            if name.rsplit("_", 1)[-1].isdigit()
            else (1, name)
        ),
    )


def json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_bddl_file(h5: h5py.File, h5_path: Path, bddl_root: Path) -> Path:
    candidates: list[Path] = []
    raw = h5["data"].attrs.get("bddl_file_name")
    if raw is not None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        source = Path(str(raw))
        candidates.extend((source, bddl_root / source.name))

    stem = h5_path.stem.removesuffix("_demo")
    candidates.append(bddl_root / f"{stem}.bddl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = list(bddl_root.rglob(f"{stem}.bddl"))
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(
        f"cannot uniquely resolve BDDL for {h5_path}; candidates={candidates}, "
        f"recursive_matches={matches}"
    )


def decode_xml(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def postprocess_demo_xml(
    xml: str,
    *,
    libero_assets_root: Path,
    base_postprocess: Any,
) -> str:
    """Make collector-machine asset paths portable before MuJoCo reload."""
    root = ET.fromstring(base_postprocess(xml))
    markers = (
        "/chiliocosm/assets/",
        "/libero/libero/assets/",
        "/libero/assets/",
    )
    for element in root.iter():
        raw = element.get("file")
        if not raw:
            continue
        normalized = raw.replace("\\", "/")
        for marker in markers:
            if marker not in normalized:
                continue
            suffix = normalized.split(marker, 1)[1]
            element.set("file", str((libero_assets_root / suffix).resolve()))
            break
    return ET.tostring(root, encoding="unicode")


def replay_demo(
    env: Any,
    demo: h5py.Group,
    *,
    postprocess_model_xml: Any,
) -> dict[str, Any]:
    if "model_file" not in demo.attrs:
        raise KeyError("demo is missing model_file")
    if "init_state" not in demo.attrs:
        raise KeyError("demo is missing init_state")
    if "actions" not in demo:
        raise KeyError("demo is missing actions")

    model_xml = postprocess_model_xml(decode_xml(demo.attrs["model_file"]))
    init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
    stored_states = (
        np.asarray(demo["states"], dtype=np.float64) if "states" in demo else None
    )
    actions = np.asarray(demo["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(f"expected non-empty actions [T,A], got {actions.shape}")

    # reset_from_xml_string recreates the exact demo model; set_init_state then
    # restores the trajectory's deterministic starting state.
    env.reset_from_xml_string(model_xml)
    env.set_init_state(init_state)

    initial_success = bool(env.check_success())
    success_any = initial_success
    first_success_step = -1
    consecutive_success = int(initial_success)
    max_consecutive_success = consecutive_success
    env_done_any = False
    env_reward_max = float("-inf")

    for step, action in enumerate(actions):
        _obs, reward, done, _info = env.step(action.tolist())
        current_success = bool(env.check_success())
        env_done_any = env_done_any or bool(done)
        env_reward_max = max(env_reward_max, float(reward))
        if current_success:
            if first_success_step < 0:
                first_success_step = step
            success_any = True
            consecutive_success += 1
            max_consecutive_success = max(
                max_consecutive_success, consecutive_success
            )
        else:
            consecutive_success = 0

    terminal_success = bool(env.check_success())
    replay_terminal_state = np.asarray(env.get_sim_state(), dtype=np.float64)
    stored_final_state_success = None
    replay_to_stored_final_state_l2 = None
    if stored_states is not None and len(stored_states):
        stored_final_state = np.asarray(stored_states[-1], dtype=np.float64)
        if stored_final_state.shape != replay_terminal_state.shape:
            raise ValueError(
                "stored/replayed terminal state shape mismatch: "
                f"{stored_final_state.shape} vs {replay_terminal_state.shape}"
            )
        replay_to_stored_final_state_l2 = float(
            np.linalg.norm(replay_terminal_state - stored_final_state)
        )
        env.set_init_state(stored_final_state)
        stored_final_state_success = bool(env.check_success())

    return {
        "num_actions": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "initial_success": initial_success,
        "success_any": success_any,
        "terminal_success": terminal_success,
        "first_success_step": int(first_success_step),
        "max_consecutive_success_steps": int(max_consecutive_success),
        "stored_final_state_success": stored_final_state_success,
        "replay_to_stored_final_state_l2": replay_to_stored_final_state_l2,
        "env_done_any": env_done_any,
        "env_reward_max": env_reward_max,
        "stored_terminal_done": (
            bool(np.asarray(demo["dones"])[-1]) if "dones" in demo else None
        ),
        "stored_terminal_reward": (
            float(np.asarray(demo["rewards"])[-1])
            if "rewards" in demo
            else None
        ),
    }


def write_outputs(
    output_dir: Path,
    report: dict[str, Any],
    verified: dict[str, list[str]],
    source_verified: dict[str, list[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(report["episodes"])
    verified_count = sum(len(names) for names in verified.values())
    source_verified_count = sum(len(names) for names in source_verified.values())
    failed = sum(item.get("status") == "failed" for item in report["episodes"])
    errors = sum(item.get("status") == "error" for item in report["episodes"])
    report["summary"] = {
        "audited": total,
        "action_replay_terminal_verified": verified_count,
        "action_replay_terminal_failed": failed,
        "stored_final_state_verified": source_verified_count,
        "errors": errors,
        "action_replay_verified_rate": verified_count / total if total else None,
        "stored_final_state_verified_rate": (
            source_verified_count / total if total else None
        ),
    }
    report["updated_at_unix"] = time.time()

    tmp_report = output_dir / "audit_report.json.tmp"
    tmp_whitelist = output_dir / "verified_demo_whitelist.json.tmp"
    tmp_source_whitelist = output_dir / "verified_source_state_whitelist.json.tmp"
    tmp_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp_whitelist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "criterion": "terminal env.check_success() after exact action replay",
                "dataset_root": report["dataset_root"],
                "suites": report["suites"],
                "num_verified": verified_count,
                "demos": verified,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp_source_whitelist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "criterion": (
                    "env.check_success() after restoring the HDF5 stored final state"
                ),
                "dataset_root": report["dataset_root"],
                "suites": report["suites"],
                "num_verified": source_verified_count,
                "demos": source_verified,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp_report.replace(output_dir / "audit_report.json")
    tmp_whitelist.replace(output_dir / "verified_demo_whitelist.json")
    tmp_source_whitelist.replace(
        output_dir / "verified_source_state_whitelist.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("LIBERO_ROOT", ROOT / "data/libero")),
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=DEFAULT_SUITES,
        help="Suite to audit; repeatable. Defaults to libero_10.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        action="append",
        help="Zero-based task id within each selected suite; repeatable.",
    )
    parser.add_argument(
        "--max-demos-per-task",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for a complete audit.",
    )
    parser.add_argument(
        "--libero-home",
        type=Path,
        default=None,
        help="Optional LIBERO repository root to prepend to sys.path.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=None,
        help="Override LIBERO assets (auto-falls back to ~/.cache/libero/assets).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/libero_expert_replay_audit",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/audit_report.json and skip audited demos.",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on the first replay error."
    )
    args = parser.parse_args()

    if args.max_demos_per_task is not None and args.max_demos_per_task <= 0:
        parser.error("--max-demos-per-task must be positive")
    suites = tuple(args.suite or ("libero_10",))
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    configure_libero(args.libero_home)
    import libero.libero as libero_package
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero.utils.utils import postprocess_model_xml

    package_bddl_root = Path(libero_package.__file__).resolve().parent / "bddl_files"
    configured_bddl_root = Path(get_libero_path("bddl_files")).expanduser()
    bddl_root = (
        package_bddl_root if package_bddl_root.is_dir() else configured_bddl_root
    ).resolve()
    libero_assets_root = resolve_assets_root(
        get_libero_path("assets"), args.assets_root
    )
    benchmark_dict = benchmark.get_benchmark_dict()
    report_path = output_dir / "audit_report.json"
    if args.resume and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if Path(report["dataset_root"]).resolve() != dataset_root:
            raise ValueError("resume dataset_root does not match the current dataset")
        if tuple(report["suites"]) != suites:
            raise ValueError("resume suites do not match the current --suite values")
        LOG.info(
            "resuming %d previously audited demos from %s",
            len(report["episodes"]),
            report_path,
        )
    else:
        report = {
            "schema_version": 1,
            "criterion": "terminal env.check_success() after exact action replay",
            "stored_hdf5_reward_done_used_for_verification": False,
            "dataset_root": str(dataset_root),
            "bddl_root": str(bddl_root),
            "libero_assets_root": str(libero_assets_root),
            "suites": list(suites),
            "seed": int(args.seed),
            "episodes": [],
            "summary": {},
        }
    verified: dict[str, list[str]] = {}
    source_verified: dict[str, list[str]] = {}
    processed = set()
    for entry in report["episodes"]:
        processed.add((entry["suite"], entry["hdf5"], entry["demo"]))
        if entry.get("terminal_success") is True:
            verified.setdefault(entry["hdf5"], []).append(entry["demo"])
        if entry.get("stored_final_state_success") is True:
            source_verified.setdefault(entry["hdf5"], []).append(entry["demo"])

    for suite_name in suites:
        suite_dir = dataset_root / suite_name
        files = sorted(suite_dir.glob("*.hdf5"))
        suite = benchmark_dict[suite_name]()
        if len(files) != suite.n_tasks:
            raise RuntimeError(
                f"{suite_name}: found {len(files)} HDF5 files but benchmark has "
                f"{suite.n_tasks} tasks"
            )
        selected_ids = args.task_id or list(range(suite.n_tasks))
        for task_id in selected_ids:
            if not 0 <= task_id < suite.n_tasks:
                raise IndexError(
                    f"task id {task_id} outside [0,{suite.n_tasks}) for {suite_name}"
                )
            task = suite.get_task(task_id)
            task_stem = Path(str(task.bddl_file)).stem
            matches = [p for p in files if p.stem.removesuffix("_demo") == task_stem]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"cannot map {suite_name} task {task_id} ({task_stem}) to one HDF5; "
                    f"matches={matches}"
                )
            h5_path = matches[0]
            relative_h5 = str(h5_path.relative_to(dataset_root))

            with h5py.File(h5_path, "r") as h5:
                bddl_path = resolve_bddl_file(h5, h5_path, bddl_root)
                demo_names = sorted_demo_names(h5)
                if args.max_demos_per_task is not None:
                    demo_names = demo_names[: args.max_demos_per_task]
                demo_names = [
                    name
                    for name in demo_names
                    if (suite_name, relative_h5, name) not in processed
                ]
                if not demo_names:
                    LOG.info(
                        "suite=%s task=%02d already complete; skipping",
                        suite_name,
                        task_id,
                    )
                    continue

                LOG.info(
                    "suite=%s task=%02d demos=%d hdf5=%s",
                    suite_name,
                    task_id,
                    len(demo_names),
                    h5_path.name,
                )
                env = ControlEnv(
                    bddl_file_name=str(bddl_path),
                    use_camera_obs=False,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    ignore_done=True,
                    horizon=max(
                        1000,
                        max(len(h5["data"][name]["actions"]) for name in demo_names)
                        + 1,
                    ),
                )
                env.seed(args.seed)
                try:
                    for demo_name in demo_names:
                        started = time.perf_counter()
                        entry: dict[str, Any] = {
                            "suite": suite_name,
                            "task_id": int(task_id),
                            "task_name": task_stem,
                            "hdf5": relative_h5,
                            "demo": demo_name,
                        }
                        try:
                            result = replay_demo(
                                env,
                                h5["data"][demo_name],
                                postprocess_model_xml=lambda xml: postprocess_demo_xml(
                                    xml,
                                    libero_assets_root=libero_assets_root,
                                    base_postprocess=postprocess_model_xml,
                                ),
                            )
                            entry.update(result)
                            entry["status"] = (
                                "verified" if result["terminal_success"] else "failed"
                            )
                            if result["terminal_success"]:
                                verified.setdefault(relative_h5, []).append(demo_name)
                            if result["stored_final_state_success"]:
                                source_verified.setdefault(relative_h5, []).append(
                                    demo_name
                                )
                        except Exception as error:
                            entry.update(
                                status="error",
                                error_type=type(error).__name__,
                                error=str(error),
                            )
                            LOG.exception(
                                "replay error suite=%s task=%d demo=%s",
                                suite_name,
                                task_id,
                                demo_name,
                            )
                            if args.fail_fast:
                                report["episodes"].append(entry)
                                write_outputs(
                                    output_dir, report, verified, source_verified
                                )
                                raise
                        entry["elapsed_seconds"] = time.perf_counter() - started
                        report["episodes"].append(entry)
                        LOG.info(
                            "suite=%s task=%02d demo=%s status=%s terminal=%s any=%s",
                            suite_name,
                            task_id,
                            demo_name,
                            entry["status"],
                            entry.get("terminal_success"),
                            entry.get("success_any"),
                        )
                        # Persist after every demo so a long audit remains resumable
                        # as evidence even if the process is interrupted.
                        write_outputs(
                            output_dir, report, verified, source_verified
                        )
                finally:
                    env.close()

    write_outputs(output_dir, report, verified, source_verified)
    summary = report["summary"]
    LOG.info(
        "audit complete replay_verified=%d/%d source_state_verified=%d "
        "replay_failed=%d errors=%d; outputs=%s",
        summary["action_replay_terminal_verified"],
        summary["audited"],
        summary["stored_final_state_verified"],
        summary["action_replay_terminal_failed"],
        summary["errors"],
        output_dir,
    )


if __name__ == "__main__":
    main()
