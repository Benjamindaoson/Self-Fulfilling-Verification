"""Paired causal metrics and group-safe resampling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def annotate_transitions(branches: pd.DataFrame) -> pd.DataFrame:
    required = {"parent_episode_id", "branch", "Y_pre", "Y_post"}
    if missing := required.difference(branches.columns):
        raise ValueError(f"missing branch columns: {sorted(missing)}")
    if (branches.groupby("parent_episode_id")["Y_pre"].nunique() != 1).any():
        raise ValueError("Y_pre differs across branches of one parent")
    controls = branches.loc[branches["branch"] == "V0", ["parent_episode_id", "Y_post"]]
    if controls["parent_episode_id"].duplicated().any() or len(controls) != branches["parent_episode_id"].nunique():
        raise ValueError("each parent must have exactly one V0 control")
    controls = controls.rename(columns={"Y_post": "control_Y_post"})
    result = branches.drop(columns=["control_Y_post"], errors="ignore").merge(
        controls, on="parent_episode_id", how="left", validate="many_to_one"
    )
    result["Y_pre"] = result["Y_pre"].astype(int)
    result["Y_post"] = result["Y_post"].astype(int)
    result["control_Y_post"] = result["control_Y_post"].astype(int)
    result["transition"] = result["Y_pre"].astype(str) + " -> " + result["Y_post"].astype(str)
    result["verification_induced_completion"] = (
        (result["Y_pre"] == 0) & (result["control_Y_post"] == 0) & (result["Y_post"] == 1)
    )
    result["verification_induced_disruption"] = (result["Y_pre"] == 1) & (result["Y_post"] == 0)
    return result


def _conditional_mean(values: pd.Series, mask: pd.Series) -> float:
    return float(values[mask].mean()) if bool(mask.any()) else float("nan")


def summarize_branch(rows: pd.DataFrame) -> dict[str, Any]:
    pre_failure = rows["Y_pre"] == 0
    pre_success = rows["Y_pre"] == 1
    post_success = rows["Y_post"] == 1
    counts = rows["transition"].value_counts()
    return {
        "n": int(len(rows)),
        "pre_failures": int(pre_failure.sum()),
        "pre_successes": int(pre_success.sum()),
        "vic": _conditional_mean(rows["verification_induced_completion"].astype(float), pre_failure),
        "vid": _conditional_mean(rows["verification_induced_disruption"].astype(float), pre_success),
        "causal_soundness": _conditional_mean(rows["Y_pre"].astype(float), post_success),
        "delta_credit": float(rows["Y_post"].mean() - rows["Y_pre"].mean()),
        "transition_counts": {label: int(counts.get(label, 0)) for label in ("0 -> 0", "0 -> 1", "1 -> 0", "1 -> 1")},
    }


def group_split(groups: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two parent groups are required")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    test_count = min(len(unique) - 1, max(1, int(round(len(unique) * test_fraction))))
    test_groups = set(shuffled[:test_count].tolist())
    is_test = np.asarray([group in test_groups for group in groups])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def bootstrap_binary(values: np.ndarray, samples: int, confidence_level: float, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    tail = (1.0 - confidence_level) / 2.0
    return {
        "estimate": float(values.mean()),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1.0 - tail)),
        "n": int(len(values)),
    }
