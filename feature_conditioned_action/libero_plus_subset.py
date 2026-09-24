"""Deterministic, stratified fast subsets for LIBERO-Plus evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


LIBERO_PLUS_CATEGORIES = (
    "Objects Layout",
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
)
LIBERO_PLUS_DIFFICULTIES = (1, 2, 3, 4, 5)


def select_official_libero_plus_tasks(
    classification_path: str | Path,
    suite_name: str,
    suite_task_names: Sequence[str],
    *,
    categories: Sequence[str] | None = None,
) -> tuple[list[int], dict[int, dict[str, Any]], dict[str, Any]]:
    """Select the official full suite or a complete official category slice.

    This function performs no subsampling. It validates the one-based official
    ids and exact suite ordering before attaching category/difficulty metadata.
    """
    if isinstance(categories, str):
        categories = (categories,)
    selected_categories = tuple(categories or LIBERO_PLUS_CATEGORIES)
    unknown = [item for item in selected_categories if item not in LIBERO_PLUS_CATEGORIES]
    if unknown:
        raise ValueError(f"unknown LIBERO-Plus categories: {unknown}")
    if len(set(selected_categories)) != len(selected_categories):
        raise ValueError("LIBERO-Plus categories must not contain duplicates")

    path = Path(classification_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if suite_name not in data:
        raise ValueError(f"classification has no suite {suite_name!r}")
    rows = data[suite_name]
    if len(rows) != len(suite_task_names):
        raise ValueError(
            f"classification/suite length mismatch for {suite_name}: "
            f"{len(rows)} vs {len(suite_task_names)}"
        )

    selected: list[int] = []
    metadata: dict[int, dict[str, Any]] = {}
    counts_by_category: dict[str, int] = defaultdict(int)
    counts_by_difficulty: dict[str, int] = defaultdict(int)
    for task_index, (row, task_name) in enumerate(zip(rows, suite_task_names)):
        if row.get("name") != task_name:
            raise ValueError(
                f"classification ordering mismatch at {suite_name}[{task_index}]: "
                f"{row.get('name')!r} != {task_name!r}"
            )
        official_id = row.get("id")
        if official_id != task_index + 1:
            raise ValueError(
                f"expected one-based official id {task_index + 1}, got {official_id!r}"
            )
        category = row.get("category")
        difficulty = row.get("difficulty_level")
        if category not in selected_categories:
            continue
        selected.append(task_index)
        metadata[task_index] = {
            "official_id": official_id,
            "category": category,
            "difficulty_level": difficulty,
            "name": task_name,
        }
        counts_by_category[str(category)] += 1
        counts_by_difficulty[str(difficulty)] += 1

    full = selected_categories == LIBERO_PLUS_CATEGORIES
    return selected, metadata, {
        "mode": "official-full" if full else "official-category",
        "official_leaderboard_subset": bool(full),
        "classification_path": str(path),
        "suite": suite_name,
        "num_tasks": len(selected),
        "categories": list(selected_categories),
        "counts_by_category": dict(counts_by_category),
        "counts_by_difficulty": dict(counts_by_difficulty),
    }


def _stable_key(seed: int, suite: str, row: dict[str, Any]) -> bytes:
    value = f"{seed}:{suite}:{row['category']}:{row['difficulty_level']}:{row['name']}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def select_balanced_libero_plus_tasks(
    classification_path: str | Path,
    suite_name: str,
    suite_task_names: Sequence[str],
    *,
    tasks_per_category: int = 10,
    seed: int = 0,
    categories: Sequence[str] | None = None,
) -> tuple[list[int], dict[int, dict[str, Any]], dict[str, Any]]:
    """Select equal category counts, spread across official difficulty levels.

    Official classification ids are one-based; evaluator suite indices are
    zero-based. Names are checked first so an upstream ordering change cannot
    silently evaluate the wrong tasks.
    """
    if tasks_per_category <= 0:
        raise ValueError("tasks_per_category must be positive")
    if isinstance(categories, str):
        categories = (categories,)
    selected_categories = tuple(categories or LIBERO_PLUS_CATEGORIES)
    if not selected_categories:
        raise ValueError("at least one LIBERO-Plus category must be selected")
    unknown_categories = [
        category for category in selected_categories
        if category not in LIBERO_PLUS_CATEGORIES
    ]
    if unknown_categories:
        raise ValueError(f"unknown LIBERO-Plus categories: {unknown_categories}")
    if len(set(selected_categories)) != len(selected_categories):
        raise ValueError("LIBERO-Plus categories must not contain duplicates")

    path = Path(classification_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if suite_name not in data:
        raise ValueError(f"classification has no suite {suite_name!r}")
    rows = data[suite_name]
    if len(rows) != len(suite_task_names):
        raise ValueError(
            f"classification/suite length mismatch for {suite_name}: "
            f"{len(rows)} vs {len(suite_task_names)}"
        )

    metadata_by_id: dict[int, dict[str, Any]] = {}
    buckets: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task_index, (row, task_name) in enumerate(zip(rows, suite_task_names)):
        if row.get("name") != task_name:
            raise ValueError(
                f"classification ordering mismatch at {suite_name}[{task_index}]: "
                f"{row.get('name')!r} != {task_name!r}"
            )
        official_id = row.get("id")
        if official_id != task_index + 1:
            raise ValueError(
                f"expected one-based official id {task_index + 1}, got {official_id!r}"
            )
        category = row.get("category")
        difficulty = row.get("difficulty_level")
        if category in LIBERO_PLUS_CATEGORIES and difficulty in LIBERO_PLUS_DIFFICULTIES:
            item = dict(row)
            item["task_index"] = task_index
            buckets[category][int(difficulty)].append(item)
        metadata_by_id[task_index] = {
            "official_id": official_id,
            "category": category,
            "difficulty_level": difficulty,
            "name": task_name,
        }

    selected: list[int] = []
    counts: dict[str, dict[str, int]] = {}
    for category in selected_categories:
        ordered = {
            difficulty: sorted(
                buckets[category][difficulty],
                key=lambda row: _stable_key(seed, suite_name, row),
            )
            for difficulty in LIBERO_PLUS_DIFFICULTIES
        }
        positions = {difficulty: 0 for difficulty in LIBERO_PLUS_DIFFICULTIES}
        category_selected: list[int] = []
        # Round-robin produces an even difficulty mix. If an official stratum is
        # sparse (some suite/category/level cells contain only one task), its
        # unfilled quota is deterministically redistributed to the other levels.
        while len(category_selected) < tasks_per_category:
            progressed = False
            for difficulty in LIBERO_PLUS_DIFFICULTIES:
                position = positions[difficulty]
                if position >= len(ordered[difficulty]):
                    continue
                category_selected.append(
                    int(ordered[difficulty][position]["task_index"])
                )
                positions[difficulty] += 1
                progressed = True
                if len(category_selected) == tasks_per_category:
                    break
            if not progressed:
                available = sum(len(rows) for rows in ordered.values())
                raise ValueError(
                    f"not enough classified tasks for {suite_name}/{category}: "
                    f"need {tasks_per_category}, found {available}"
                )
        selected.extend(category_selected)
        for difficulty in LIBERO_PLUS_DIFFICULTIES:
            counts.setdefault(category, {})[str(difficulty)] = positions[difficulty]

    selected.sort()
    all_categories = selected_categories == LIBERO_PLUS_CATEGORIES
    summary = {
        "mode": "balanced-fast" if all_categories else "category-balanced-fast",
        "official_leaderboard_subset": False,
        "classification_path": str(path),
        "suite": suite_name,
        "seed": int(seed),
        "tasks_per_category": int(tasks_per_category),
        "num_tasks": len(selected),
        "categories": list(selected_categories),
        "difficulty_levels": list(LIBERO_PLUS_DIFFICULTIES),
        "counts_by_category_and_difficulty": counts,
    }
    return selected, metadata_by_id, summary


def summarize_classified_results(
    task_results: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Aggregate evaluated episodes by official perturbation and difficulty."""
    grouped: dict[str, dict[str, dict[str, int]]] = {
        "by_category": {},
        "by_difficulty": {},
    }
    classified_tasks = 0
    for result in task_results.values():
        metadata = result.get("libero_plus_classification")
        if not metadata:
            continue
        classified_tasks += 1
        successes = sum(int(episode["success"]) for episode in result["episodes"])
        trials = len(result["episodes"])
        keys = {
            "by_category": str(metadata["category"]),
            "by_difficulty": str(metadata["difficulty_level"]),
        }
        for group_name, key in keys.items():
            item = grouped[group_name].setdefault(
                key, {"successes": 0, "trials": 0, "num_tasks": 0}
            )
            item["successes"] += successes
            item["trials"] += trials
            item["num_tasks"] += 1
    if not classified_tasks:
        return None
    for group in grouped.values():
        for item in group.values():
            item["success_rate"] = item["successes"] / max(item["trials"], 1)
    return {"classified_tasks": classified_tasks, **grouped}
