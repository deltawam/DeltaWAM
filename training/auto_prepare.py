"""Best-effort automatic preparation of external training assets.

This module keeps heavyweight downloads out of the model/dataset code while
still making first-run training less brittle:

* missing ActionDiT Wan2.2 init payload -> download/cache Wan2.2 and build it;
* missing LIBERO hdf5 suite(s) -> download and extract configured suite data.

All behavior is opt-out via environment variables and is intentionally explicit
about where files are stored.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlretrieve

import torch


WAN22_DEFAULT_REPO = "Wan-AI/Wan2.2-TI2V-5B"

DEFAULT_LIBERO_URLS = {
    "libero_spatial": "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip",
    "libero_object": "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip",
    "libero_goal": "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip",
    "libero_90": "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip",
    "libero_10": "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip",
}
DEFAULT_LIBERO_ZIP_NAMES = {
    "libero_spatial": "libero_spatial.zip",
    "libero_object": "libero_object.zip",
    "libero_goal": "libero_goal.zip",
    "libero_90": "libero_100.zip",
    "libero_10": "libero_100.zip",
}

KINETICS_CKPT_SPECS = {
    "ckpt/deltatok-kinetics/pytorch_model.bin": (
        "Amazon-FAR/deltatok-kinetics",
        "pytorch_model.bin",
    ),
    "ckpt/deltaworld-kinetics/pytorch_model.bin": (
        "Amazon-FAR/deltaworld-kinetics",
        "pytorch_model.bin",
    ),
}


def ensure_kinetics_ckpt(path: str | os.PathLike[str]) -> str:
    """Return a local checkpoint path, downloading known release ckpts only if missing."""

    resolved = _resolve_path(path)
    if resolved.is_file():
        return str(resolved)

    rel_key = resolved.relative_to(_repo_root()).as_posix() if resolved.is_relative_to(_repo_root()) else str(path)
    spec = KINETICS_CKPT_SPECS.get(rel_key)
    if spec is None:
        if not _env_flag("DELTATOK_AUTO_DOWNLOAD_KINETICS_CKPT", True):
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved}. Automatic download is only configured for "
            f"{sorted(KINETICS_CKPT_SPECS)}."
        )

    if not _env_flag("DELTATOK_AUTO_DOWNLOAD_KINETICS_CKPT", True):
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved}. Set DELTATOK_AUTO_DOWNLOAD_KINETICS_CKPT=1 "
            "or download it manually."
        )

    repo_id, filename = spec
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "huggingface_hub is required for automatic DeltaTok/DeltaWorld checkpoint "
            "downloads. Install it or pre-populate the required ckpt files."
        ) from exc

    resolved.parent.mkdir(parents=True, exist_ok=True)
    print(f"[auto-prepare] Downloading {repo_id}/{filename} -> {resolved}")
    downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    tmp = resolved.with_suffix(resolved.suffix + ".tmp")
    shutil.copyfile(downloaded, tmp)
    tmp.replace(resolved)
    return str(resolved)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _repo_root() / candidate


def _snapshot_download(repo_id: str, local_dir: Path, *, repo_type: str | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "huggingface_hub is required for automatic HuggingFace downloads. "
            "Install it or pre-populate the required files."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
    }
    if repo_type is not None:
        kwargs["repo_type"] = repo_type
    return Path(snapshot_download(**kwargs))


def ensure_wan22_model_dir() -> Path:
    """Return a local Wan2.2 model directory, downloading it if necessary."""

    for env_name in ("WAN22_MODEL_DIR", "WAN2_2_MODEL_DIR", "WAN_MODEL_DIR"):
        value = os.environ.get(env_name)
        if value:
            path = Path(value).expanduser()
            if path.exists():
                return path

    if not _env_flag("DELTATOK_AUTO_DOWNLOAD_WAN22", True):
        raise FileNotFoundError(
            "Wan2.2 model directory is missing and DELTATOK_AUTO_DOWNLOAD_WAN22=0. "
            "Set WAN22_MODEL_DIR or pre-generate action_init_path."
        )

    repo_id = os.environ.get("WAN22_HF_REPO", WAN22_DEFAULT_REPO)
    cache_dir = Path(
        os.environ.get(
            "WAN22_CACHE_DIR",
            str(_repo_root() / "ckpt" / "Wan-AI" / "Wan2.2-TI2V-5B"),
        )
    ).expanduser()
    print(f"[auto-prepare] Downloading Wan2.2 from HuggingFace repo {repo_id} -> {cache_dir}")
    return _snapshot_download(repo_id, cache_dir)


def ensure_action_dit_init_payload(
    init_from: str | None,
    action_dit: torch.nn.Module,
    *,
    head_init: str = "random",
    dtype: torch.dtype = torch.bfloat16,
) -> str | None:
    """Ensure a configured ActionDiT init payload exists.

    If ``init_from`` is ``None`` the caller explicitly requested random
    initialization, so this function leaves it alone. If a path is configured
    but missing, it builds the payload from Wan2.2.
    """

    if not init_from:
        return None
    output = _resolve_path(init_from)
    if output.is_file():
        return str(output)
    if not _env_flag("DELTATOK_AUTO_PREPARE_ACTION_INIT", True):
        raise FileNotFoundError(f"ActionDiT init payload not found: {output}")

    # Reuse the project's preprocessing implementation to keep mapping semantics
    # identical between manual and automatic payload creation.
    from scripts.preprocess_feature_action_dit_init import (
        convert_tensor,
        load_wan22_video_state,
        maybe_write_json_summary,
    )

    wan_dir = ensure_wan22_model_dir()
    print(f"[auto-prepare] Building ActionDiT init payload {output} from {wan_dir}")
    video_state = load_wan22_video_state(str(wan_dir), dtype=dtype)
    action_state = action_dit.state_dict()
    backbone_keys = sorted(action_dit.backbone_key_set(action_state.keys()))
    mapped: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    copied = interpolated = 0
    shape_changes: dict[str, dict[str, list[int]]] = {}
    for key in backbone_keys:
        if key not in video_state:
            missing.append(key)
            continue
        value, resized = convert_tensor(video_state[key], action_state[key], alpha_scaling=True)
        mapped[key] = value
        if resized:
            interpolated += 1
            shape_changes[key] = {
                "source": list(video_state[key].shape),
                "target": list(action_state[key].shape),
            }
        else:
            copied += 1
    if not mapped:
        raise RuntimeError(
            f"Failed to map any Wan2.2 weights into ActionDiT from {wan_dir}. "
            "Check WAN22_MODEL_DIR/WAN22_HF_REPO."
        )

    payload: dict[str, Any] = {
        "policy": {
            "source": "auto_wan22_model_dir",
            "source_backbone": "wan22",
            "alpha_scaling": True,
            "interpolation": "sequential_1d_linear_align_corners_true",
            "action_backbone_skip_prefixes": list(action_dit.ACTION_BACKBONE_SKIP_PREFIXES),
            "head_init": head_init,
            "missing_allowed": True,
        },
        "backbone_state_dict": mapped,
        "meta": {
            "mapped_keys": len(mapped),
            "missing_keys": len(missing),
            "auto_prepared": True,
            "wan22_model_dir": str(wan_dir),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    maybe_write_json_summary(
        output,
        payload,
        {
            "total_action_backbone_keys": len(backbone_keys),
            "mapped": len(mapped),
            "copied": copied,
            "interpolated": interpolated,
            "missing": missing,
            "shape_changes": shape_changes,
        },
    )
    print(
        f"[auto-prepare] Saved ActionDiT init payload to {output} "
        f"(mapped={len(mapped)}/{len(backbone_keys)}, missing={len(missing)})"
    )
    return str(output)


def _suite_has_hdf5(root: Path, suite: str) -> bool:
    return any((root / suite).glob("*.hdf5"))


def _suite_env_name(prefix: str, suite: str) -> str:
    return f"{prefix}_{suite.upper()}"


def _download_url(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    print(f"[auto-prepare] Downloading {url} -> {dst}")
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    urlretrieve(url, tmp)
    tmp.replace(dst)
    return dst


def _extract_zip(zip_path: Path, root: Path) -> None:
    print(f"[auto-prepare] Extracting {zip_path} -> {root}")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)


def _try_hf_dataset(root: Path, suite: str) -> bool:
    repo = os.environ.get("LIBERO_HF_DATASET_REPO")
    if not repo:
        return False
    cache_dir = Path(os.environ.get("LIBERO_HF_CACHE_DIR", str(root / "_hf_download"))).expanduser()
    print(f"[auto-prepare] Downloading LIBERO dataset repo {repo} for missing suite {suite}")
    local = _snapshot_download(repo, cache_dir, repo_type="dataset")
    for candidate in (local / suite, local / "data" / "libero" / suite):
        if candidate.exists() and candidate != root / suite:
            (root / suite).parent.mkdir(parents=True, exist_ok=True)
            if not (root / suite).exists():
                try:
                    os.symlink(candidate, root / suite, target_is_directory=True)
                except OSError:
                    shutil.copytree(candidate, root / suite)
            return _suite_has_hdf5(root, suite)
    zip_candidates = list(local.glob(f"**/{suite}.zip"))
    if suite == "libero_90":
        zip_candidates += list(local.glob("**/libero_100.zip"))
    for zip_path in zip_candidates:
        _extract_zip(zip_path, root)
        if _suite_has_hdf5(root, suite):
            return True
    return False


def _try_kaggle(root: Path, suite: str) -> bool:
    handle = os.environ.get("LIBERO_KAGGLE_HANDLE")
    if not handle:
        return False
    try:
        import kagglehub
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("kagglehub is required for LIBERO_KAGGLE_HANDLE downloads") from exc
    print(f"[auto-prepare] Downloading LIBERO Kaggle dataset {handle}")
    local = Path(kagglehub.dataset_download(handle))
    for candidate in (local / suite, local / "data" / "libero" / suite):
        if candidate.exists() and not (root / suite).exists():
            shutil.copytree(candidate, root / suite)
    for zip_path in list(local.glob(f"**/{suite}.zip")) + list(local.glob("**/libero_100.zip")):
        _extract_zip(zip_path, root)
    return _suite_has_hdf5(root, suite)


def ensure_libero_suites(root: Path, suites: Iterable[str]) -> None:
    """Ensure requested LIBERO suites exist under ``root``.

    Download sources, in priority order:
      1. direct URL env: LIBERO_URL_LIBERO_SPATIAL, ...;
      2. LIBERO_DOWNLOAD_BASE_URL + ``<zip_name>``;
      3. built-in official LIBERO Box URLs;
      4. LIBERO_HF_DATASET_REPO snapshot;
      5. LIBERO_KAGGLE_HANDLE.
    """

    missing = [suite for suite in suites if not _suite_has_hdf5(root, suite)]
    if not missing:
        return
    if not _env_flag("DELTATOK_AUTO_DOWNLOAD_LIBERO", True):
        raise FileNotFoundError(
            f"Missing LIBERO suites under {root}: {missing}. "
            "Set DELTATOK_AUTO_DOWNLOAD_LIBERO=1 or populate the data manually."
        )

    root.mkdir(parents=True, exist_ok=True)
    base_url = os.environ.get("LIBERO_DOWNLOAD_BASE_URL", "").rstrip("/")
    for suite in missing:
        if _suite_has_hdf5(root, suite):
            continue
        zip_name = DEFAULT_LIBERO_ZIP_NAMES.get(suite, f"{suite}.zip")
        url = os.environ.get(_suite_env_name("LIBERO_URL", suite))
        if url is None and base_url:
            url = f"{base_url}/{zip_name}"
        if url is None:
            url = DEFAULT_LIBERO_URLS.get(suite)
        if url:
            zip_path = root / zip_name
            _download_url(url, zip_path)
            _extract_zip(zip_path, root)
        if not _suite_has_hdf5(root, suite) and _try_hf_dataset(root, suite):
            continue
        if not _suite_has_hdf5(root, suite) and _try_kaggle(root, suite):
            continue
        if not _suite_has_hdf5(root, suite):
            raise FileNotFoundError(
                f"Could not automatically prepare LIBERO suite {suite!r} under {root}. "
                "The built-in LIBERO official URLs failed or did not contain the expected hdf5 files. "
                "You can override with LIBERO_URL_<SUITE>, LIBERO_DOWNLOAD_BASE_URL, "
                "LIBERO_HF_DATASET_REPO, or LIBERO_KAGGLE_HANDLE."
            )


def clean_partial_processes() -> None:
    """Best-effort helper for callers that need to avoid stale workers."""

    # Intentionally not used automatically; kept as a tiny utility for scripts.
    subprocess.run(["true"], check=False)
