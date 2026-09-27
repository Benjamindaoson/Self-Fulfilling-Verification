"""Analyze a saved Gate-0 run without rerunning simulation."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gate0.config import config_hash, load_config
from gate0.metrics import annotate_transitions, bootstrap_binary, group_split, summarize_branch


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_file(run_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _verify_run_integrity(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Fail closed before analysis writes if any frozen simulator input changed."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config_hash(config) != manifest.get("config_hash"):
        raise RuntimeError("config hash mismatch: frozen run configuration changed")

    ledger_hashes = manifest.get("input_ledger_hashes")
    if not isinstance(ledger_hashes, dict):
        raise RuntimeError("manifest is missing immutable input ledger hashes")
    expected_ledgers = {
        "results/parent_rollouts.parquet",
        "results/verification_branches.parquet",
        "results/trajectory_index.parquet",
    }
    if set(ledger_hashes) != expected_ledgers:
        raise RuntimeError("manifest does not enumerate the three immutable input ledgers")
    for relative, expected in ledger_hashes.items():
        path = _run_file(run_dir, relative)
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"input ledger hash mismatch: {relative}")

    trajectory_index_path = run_dir / "results" / "trajectory_index.parquet"
    trajectories = pd.read_parquet(trajectory_index_path)
    for row in trajectories.itertuples(index=False):
        path = _run_file(run_dir, row.trajectory_path)
        if not path.is_file() or _sha256(path) != row.sha256:
            raise RuntimeError(f"trajectory hash mismatch: {row.trajectory_path}")

    parents = pd.read_parquet(run_dir / "results" / "parent_rollouts.parquet")
    for row in parents.itertuples(index=False):
        path = _run_file(run_dir, row.snapshot_path)
        if not path.is_file():
            raise RuntimeError(f"snapshot missing: {row.snapshot_path}")
        snapshot = np.load(path, allow_pickle=False)
        if hashlib.sha256(snapshot.tobytes()).hexdigest() != row.snapshot_hash:
            raise RuntimeError(f"snapshot hash mismatch: {row.snapshot_path}")
    return manifest


def _analysis_provenance(run_dir: Path) -> dict[str, Any]:
    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=run_dir, capture_output=True, text=True, check=False
    )
    if root_result.returncode != 0:
        raise RuntimeError("analysis must run inside the source Git repository")
    repo_root = Path(root_result.stdout.strip()).resolve()

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed during analysis provenance")
        return result.stdout.strip()

    if git("status", "--porcelain=v1"):
        raise RuntimeError("refusing analysis from a dirty Git worktree")
    lock_path = repo_root / "requirements.lock.txt"
    if not lock_path.is_file():
        raise RuntimeError("requirements.lock.txt is required for analysis")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": False,
        "dependency_lock_sha256": _sha256(lock_path),
        "python_version": platform.python_version(),
        "dependency_versions": {
            package: version(package)
            for package in ("matplotlib", "mujoco", "numpy", "pandas", "pyarrow", "PyYAML", "scikit-learn", "scipy")
        },
    }


def _as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    return array[:, None] if array.ndim == 1 else array.reshape(len(array), -1)


def _load_trace(path: str, run_dir: Path) -> dict[str, np.ndarray]:
    trace_path = Path(path)
    if not trace_path.is_absolute():
        trace_path = run_dir / trace_path
    with np.load(trace_path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _resample(matrix: np.ndarray, steps: int) -> np.ndarray:
    matrix = _as_2d(matrix)
    old = np.linspace(0.0, 1.0, len(matrix))
    new = np.linspace(0.0, 1.0, steps)
    return np.column_stack([np.interp(new, old, matrix[:, column]) for column in range(matrix.shape[1])])


def _trace_features(trace: dict[str, np.ndarray], resample_steps: int, timestep: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    final = np.concatenate(
        [
            np.asarray(trace["peg_pose"][-1], dtype=float),
            np.asarray(trace["relative_pose"][-1], dtype=float),
            np.asarray(
                [
                    trace["insertion_depth"][-1],
                    trace["lateral_error"][-1],
                    trace["angular_error_deg"][-1],
                    trace["contact_count"][-1],
                    trace["contact_force"][-1],
                    trace["contact_penetration"][-1],
                ],
                dtype=float,
            ),
        ]
    )
    velocity = np.linalg.norm(_as_2d(trace["qvel"])[:, :3], axis=1)
    command = np.linalg.norm(_as_2d(trace["actuator_command"])[:, :3], axis=1)
    contact = np.asarray(trace["contact_count"], dtype=float) > 0
    position = _as_2d(trace["peg_pose"])[:, :3]
    path_length = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()) if len(position) > 1 else 0.0
    contact_onset = float(np.argmax(contact) / max(1, len(contact) - 1)) if contact.any() else 1.0
    summary_series = np.column_stack(
        [
            trace["insertion_depth"],
            trace["lateral_error"],
            trace["angular_error_deg"],
            trace["contact_count"],
            trace["contact_force"],
            trace["contact_penetration"],
            velocity,
            command,
        ]
    ).astype(float)
    summary = np.concatenate(
        [
            summary_series[0],
            summary_series[-1],
            summary_series[-1] - summary_series[0],
            np.asarray(
                [
                    np.max(trace["contact_force"]),
                    np.mean(trace["contact_force"]),
                    np.sum(trace["contact_force"][1:]) * timestep,
                    np.max(trace["contact_penetration"]),
                    path_length,
                    contact_onset,
                    np.mean(contact),
                    np.max(velocity),
                    np.mean(velocity),
                    np.max(command),
                    np.mean(command),
                ]
            ),
        ]
    )
    trace_matrix = np.column_stack(
        [
            _as_2d(trace["qpos"]),
            _as_2d(trace["qvel"]),
            _as_2d(trace["relative_pose"]),
            _as_2d(trace["contact_count"]),
            _as_2d(trace["contact_force"]),
            _as_2d(trace["force_torque"]),
            _as_2d(trace["contact_penetration"]),
            _as_2d(trace["actuator_command"]),
            _as_2d(trace["verification_action"]),
            _as_2d(trace["insertion_depth"]),
            _as_2d(trace["lateral_error"]),
            _as_2d(trace["angular_error_deg"]),
            _as_2d(trace["success"]),
        ]
    )
    full_trace = _resample(trace_matrix, resample_steps).ravel()
    return final, summary, full_trace


def _feature_sets(branches: pd.DataFrame, config: dict[str, Any], run_dir: Path) -> dict[str, np.ndarray]:
    final, summary, full = [], [], []
    for path in branches["trajectory_path"]:
        features = _trace_features(
            _load_trace(path, run_dir),
            int(config["analysis"]["resample_steps"]),
            float(config["simulation"]["timestep"]),
        )
        final.append(features[0])
        summary.append(features[1])
        full.append(features[2])
    return {"R1_final": np.vstack(final), "R2_summary": np.vstack(summary), "R3_full_trace": np.vstack(full)}


def _models(seed: int) -> dict[str, Any]:
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        ),
        "MLPClassifier": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), solver="lbfgs", max_iter=350, random_state=seed)
        ),
    }


def _bootstrap_balanced(y_true: np.ndarray, y_pred: np.ndarray, samples: int, confidence: float, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)

    def score(indices: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    np.mean(y_pred[indices][y_true[indices] == label] == label)
                    for label in classes
                    if np.any(y_true[indices] == label)
                ]
            )
        )

    scores = []
    for _ in range(samples):
        index = np.concatenate(
            [rng.choice(np.flatnonzero(y_true == label), size=int((y_true == label).sum()), replace=True) for label in classes]
        )
        scores.append(score(index))
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": score(np.arange(len(y_true))),
        "low": float(np.quantile(scores, tail)),
        "high": float(np.quantile(scores, 1.0 - tail)),
        "n": int(len(y_true)),
    }


def _recoverability(
    branches: pd.DataFrame, features: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    analysis = config["analysis"]
    groups = branches["parent_episode_id"].to_numpy()
    train_index, test_index = group_split(groups, float(analysis["test_fraction"]), int(analysis["split_seed"]))
    train_groups, test_groups = set(groups[train_index]), set(groups[test_index])
    if not train_groups.isdisjoint(test_groups):
        raise RuntimeError("parent leakage in train/test split")
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    falsification_max: dict[str, Any] = {}
    for branch_number, branch_id in enumerate(config["verification"]["branches"]):
        branch_mask = branches["branch"].to_numpy() == branch_id
        branch_train = branch_mask & np.asarray([group in train_groups for group in groups])
        branch_test = branch_mask & np.asarray([group in test_groups for group in groups])
        falsification_max[branch_id] = {}
        for target_name, target_column in (("Y_pre", "Y_pre"), ("Y_post", "Y_post")):
            y = branches[target_column].to_numpy(dtype=int)
            falsification_max[branch_id][target_name] = {}
            for regime_number, (regime, x) in enumerate(features.items()):
                suite_records: list[dict[str, Any]] = []
                y_train, y_test = y[branch_train], y[branch_test]
                class_counts = {
                    "train_negative_n": int((y_train == 0).sum()),
                    "train_positive_n": int((y_train == 1).sum()),
                    "test_negative_n": int((y_test == 0).sum()),
                    "test_positive_n": int((y_test == 1).sum()),
                }
                majority = int(y_train.mean() >= 0.5)
                majority_prediction = np.full(len(y_test), majority, dtype=int)
                baseline_seed = int(analysis["classifier_seed"]) + branch_number * 100 + regime_number * 10 + 9
                baseline_ci = _bootstrap_balanced(
                    y_test, majority_prediction, int(analysis["bootstrap_samples"]),
                    float(analysis["confidence_level"]), baseline_seed,
                )
                records.append({
                    "branch": branch_id,
                    "verification_strength": float(config["verification"]["branches"][branch_id]["strength"]),
                    "target": target_name,
                    "regime": regime,
                    "model": "MajorityBaseline",
                    "train_n": int(branch_train.sum()),
                    "test_n": int(branch_test.sum()),
                    **class_counts,
                    "status": "OK",
                    "accuracy": float(accuracy_score(y_test, majority_prediction)),
                    "balanced_accuracy": baseline_ci["estimate"],
                    "roc_auc": None,
                    "ci": baseline_ci,
                    "test_positive_rate": float(y_test.mean()),
                    "predicted_positive_rate": float(majority_prediction.mean()),
                })
                test_rows = branches.loc[branch_test]
                for row_number, (row_index, row) in enumerate(test_rows.iterrows()):
                    predictions.append({
                        "row_index": int(row_index), "parent_episode_id": row["parent_episode_id"],
                        "seed": int(row["seed"]), "branch": branch_id, "target": target_name,
                        "regime": regime, "model": "MajorityBaseline", "y_true": int(y_test[row_number]),
                        "y_pred": majority, "probability": float(majority),
                    })
                for model_number, (model_name, model) in enumerate(_models(int(analysis["classifier_seed"])).items()):
                    base = {
                        "branch": branch_id,
                        "verification_strength": float(
                            config["verification"]["branches"][branch_id]["strength"]
                        ),
                        "target": target_name,
                        "regime": regime,
                        "model": model_name,
                        "train_n": int(branch_train.sum()),
                        "test_n": int(branch_test.sum()),
                        **class_counts,
                    }
                    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                        record = {**base, "status": "N/A_SINGLE_CLASS", "accuracy": None, "balanced_accuracy": None, "roc_auc": None, "ci": None}
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            model.fit(x[branch_train], y_train)
                        prediction = model.predict(x[branch_test]).astype(int)
                        probability = model.predict_proba(x[branch_test])[:, 1]
                        auc = float(roc_auc_score(y_test, probability)) if len(np.unique(y_test)) == 2 else None
                        ci_seed = int(analysis["classifier_seed"]) + branch_number * 100 + regime_number * 10 + model_number
                        record = {
                            **base,
                            "status": "OK",
                            "accuracy": float(accuracy_score(y_test, prediction)),
                            "balanced_accuracy": _bootstrap_balanced(y_test, prediction, 1, 0.95, ci_seed)["estimate"],
                            "roc_auc": auc,
                            "ci": _bootstrap_balanced(
                                y_test,
                                prediction,
                                int(analysis["bootstrap_samples"]),
                                float(analysis["confidence_level"]),
                                ci_seed,
                            ),
                            "test_positive_rate": float(y_test.mean()),
                            "predicted_positive_rate": float(prediction.mean()),
                        }
                        for row_number, (row_index, row) in enumerate(test_rows.iterrows()):
                            predictions.append({
                                "row_index": int(row_index), "parent_episode_id": row["parent_episode_id"],
                                "seed": int(row["seed"]), "branch": branch_id, "target": target_name,
                                "regime": regime, "model": model_name, "y_true": int(y_test[row_number]),
                                "y_pred": int(prediction[row_number]), "probability": float(probability[row_number]),
                            })
                    records.append(record)
                    if record["status"] == "OK":
                        suite_records.append(record)
                falsification_max[branch_id][target_name][regime] = (
                    max(suite_records, key=lambda item: (item["balanced_accuracy"], item["accuracy"]))
                    if suite_records else None
                )
    split = {
        "train_parent_count": len(train_groups),
        "test_parent_count": len(test_groups),
        "overlap": sorted(train_groups.intersection(test_groups)),
    }
    assignments = [
        {"parent_episode_id": group, "split": "train" if group in train_groups else "test"}
        for group in sorted(set(groups))
    ]
    return records, {
        "falsification_rule": analysis["falsification_rule"],
        "falsification_max": falsification_max,
        "split": split,
    }, predictions, assignments


def _branch_metrics(branches: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries, bootstrap = {}, {}
    samples = int(config["analysis"]["bootstrap_samples"])
    confidence = float(config["analysis"]["confidence_level"])
    for index, branch_id in enumerate(config["verification"]["branches"]):
        rows = branches[branches["branch"] == branch_id]
        summary = summarize_branch(rows)
        pre_failure = rows["Y_pre"] == 0
        pre_success = rows["Y_pre"] == 1
        post_success = rows["Y_post"] == 1
        near = pre_failure & (rows["pre_success_margin"] >= -1.0) & (rows["pre_success_margin"] < 0.0)
        seed = int(config["analysis"]["split_seed"]) + index * 10
        intervals = {
            "vic": bootstrap_binary(rows.loc[pre_failure, "verification_induced_completion"], samples, confidence, seed),
            "vic_near_success": bootstrap_binary(
                rows.loc[near, "verification_induced_completion"], samples, confidence, seed + 4
            ),
            "vid": bootstrap_binary(rows.loc[pre_success, "verification_induced_disruption"], samples, confidence, seed + 1),
            "causal_soundness": bootstrap_binary(rows.loc[post_success, "Y_pre"], samples, confidence, seed + 2),
            "delta_credit": bootstrap_binary((rows["Y_post"] - rows["Y_pre"]).to_numpy(), samples, confidence, seed + 3),
        }
        summary["vic_near_success"] = intervals["vic_near_success"]["estimate"]
        summary["control_stable_pre_failures"] = int((pre_failure & (rows["control_Y_post"] == 0)).sum())
        summary["verification_strength"] = float(rows["verification_strength"].iloc[0])
        summaries[branch_id], bootstrap[branch_id] = summary, intervals
    return summaries, bootstrap


def _naturalness(branches: pd.DataFrame, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bins = config["analysis"]["margin_bins"]
    labels = config["analysis"]["margin_labels"]
    rows = branches.copy()
    rows["margin_bin"] = pd.cut(rows["pre_success_margin"], bins=bins, labels=labels, right=False)
    table = []
    for branch_id in config["verification"]["branches"]:
        for label in labels:
            subset = rows[(rows["branch"] == branch_id) & (rows["margin_bin"] == label) & (rows["Y_pre"] == 0)]
            table.append(
                {
                    "branch": branch_id,
                    "margin_bin": label,
                    "n": int(len(subset)),
                    "completion_count": int(subset["verification_induced_completion"].sum()),
                    "vic": float(subset["verification_induced_completion"].mean()) if len(subset) else None,
                }
            )
    strongest = rows[(rows["branch"] == "V4") & rows["verification_induced_completion"]]
    very_near = int((strongest["margin_bin"] == "very_near_boundary").sum())
    audit = {
        "strong_completion_count": int(len(strongest)),
        "very_near_completion_count": very_near,
        "very_near_fraction": float(very_near / len(strongest)) if len(strongest) else None,
        "non_very_near_completion_count": int(len(strongest) - very_near),
    }
    return table, audit


def _value(decision: dict[str, Any], branch: str, target: str, regime: str) -> float | None:
    record = decision[branch][target][regime]
    return None if record is None else float(record["balanced_accuracy"])


def _decide(
    branches: pd.DataFrame,
    branch_metrics: dict[str, Any],
    recovery: dict[str, Any],
    naturalness: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, dict[str, bool], dict[str, Any], list[str]]:
    gate = config["gate"]
    falsification_max = recovery["falsification_max"]
    vic = {branch: branch_metrics[branch]["vic"] for branch in ("V2", "V3", "V4")}
    cs = {branch: branch_metrics[branch]["causal_soundness"] for branch in ("V2", "V3", "V4")}
    pre = {branch: _value(falsification_max, branch, "Y_pre", "R3_full_trace") for branch in ("V2", "V3", "V4")}
    post = {branch: _value(falsification_max, branch, "Y_post", "R3_full_trace") for branch in ("V2", "V3", "V4")}
    h1 = bool(vic["V4"] is not None and vic["V4"] >= float(gate["min_vic"]))
    h2 = bool(
        vic["V2"] <= vic["V3"] <= vic["V4"]
        and vic["V4"] - vic["V2"] >= float(gate["min_vic_increase"])
    )
    h3 = bool(
        all(value is not None and value >= float(gate["min_post_recoverability"]) for value in post.values())
        and post["V4"] >= post["V2"] - 0.03
    )
    pre_drop = pre["V2"] is not None and pre["V4"] is not None and pre["V2"] - pre["V4"] >= float(
        gate["min_pre_recoverability_drop"]
    )
    cs_drop = (
        cs["V2"] is not None
        and cs["V4"] is not None
        and np.isfinite(cs["V2"])
        and np.isfinite(cs["V4"])
        and cs["V2"] - cs["V4"] >= float(gate["min_pre_recoverability_drop"])
    )
    h4 = bool(pre_drop or cs_drop)
    h5 = bool(pre["V4"] is not None and pre["V4"] < float(gate["max_full_trace_pre_recoverability"]))
    hypotheses = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5}

    ypre_consistent = bool((branches.groupby("parent_episode_id")["Y_pre"].nunique() == 1).all())
    hash_consistent = bool((branches.groupby("parent_episode_id")["initial_state_hash"].nunique() == 1).all())
    v0 = branches[(branches["branch"] == "V0") & (branches["Y_pre"] == 0)]
    v0_completion = float(v0["Y_post"].mean()) if len(v0) else 0.0
    mean_depth_effect = {
        branch: float(
            (branches.loc[branches["branch"] == branch, "post_insertion_depth"] - branches.loc[branches["branch"] == branch, "pre_insertion_depth"]).mean()
        )
        for branch in ("V2", "V3", "V4")
    }
    physical_distinctness = bool(
        mean_depth_effect["V2"] < mean_depth_effect["V3"] < mean_depth_effect["V4"]
        and mean_depth_effect["V4"] - mean_depth_effect["V2"] > 1e-4
    )
    natural = bool(
        naturalness["very_near_fraction"] is not None
        and naturalness["very_near_fraction"] <= float(gate["max_very_near_vic_fraction"])
        and naturalness["non_very_near_completion_count"] > 0
    )
    plausible = bool(vic["V4"] <= float(gate["max_strong_probe_vic"]))
    sanity = {
        "check_A_y_pre_identical": ypre_consistent,
        "check_B_initial_hash_identical": hash_consistent,
        "check_C_v0_completion_rate": v0_completion,
        "check_C_pass": v0_completion <= float(gate["max_v0_completion"]),
        "check_D_mean_depth_effect": mean_depth_effect,
        "check_D_pass": physical_distinctness,
        "check_E_strong_probe_vic": vic["V4"],
        "check_E_pass": plausible,
        "naturalness_pass": natural,
    }
    reasons = []
    if not h1:
        status = "KILL"
        reasons.append("Natural verification-induced completion did not reach the frozen minimum.")
    elif not sanity["check_C_pass"]:
        status = "KILL"
        reasons.append("The no-action control completed too many pre-failures; task termination is invalid.")
    elif not physical_distinctness:
        status = "KILL"
        reasons.append("Probe strengths did not produce physically distinct insertion effects.")
    elif not plausible:
        status = "KILL"
        reasons.append("The strongest probe acted as near-universal recovery rather than plausible verification.")
    elif not natural:
        status = "KILL"
        reasons.append("Completions were absent outside the frozen very-near-boundary layer.")
    elif not h5:
        status = "KILL"
        reasons.append("At least one predeclared full-trace classifier nearly recovered Y_pre.")
    elif all(hypotheses.values()):
        status = "GO"
        reasons.append("All five frozen hypotheses and all validity/naturalness checks passed.")
    else:
        status = "CONTINUE"
        reasons.append("Natural causal miscredit exists, but the stronger history-degradation claim did not clear every gate.")
    return status, hypotheses, sanity, reasons


def _errorbar(ax: Any, x: np.ndarray, records: list[dict[str, Any] | None], label: str, marker: str) -> None:
    estimates = np.asarray([np.nan if record is None else record["estimate"] for record in records], dtype=float)
    low = np.asarray([np.nan if record is None else record["low"] for record in records], dtype=float)
    high = np.asarray([np.nan if record is None else record["high"] for record in records], dtype=float)
    ax.errorbar(x, estimates, yerr=np.vstack([estimates - low, high - estimates]), marker=marker, capsize=3, label=label)


def _figures(
    run_dir: Path,
    parents: pd.DataFrame,
    branches: pd.DataFrame,
    branch_metrics: dict[str, Any],
    bootstrap: dict[str, Any],
    recovery: dict[str, Any],
    naturalness_table: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    figures = run_dir / "figures"
    branch_ids = list(config["verification"]["branches"])
    strength = np.asarray([config["verification"]["branches"][branch]["strength"] for branch in branch_ids], dtype=float)
    falsification_max = recovery["falsification_max"]
    post_records = [falsification_max[branch]["Y_post"]["R3_full_trace"] for branch in branch_ids]
    pre_records = [falsification_max[branch]["Y_pre"]["R3_full_trace"] for branch in branch_ids]
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    _errorbar(ax, strength, [bootstrap[branch]["vic"] for branch in branch_ids], "VIC", "o")
    _errorbar(ax, strength, [bootstrap[branch]["causal_soundness"] for branch in branch_ids], "Causal soundness", "s")
    _errorbar(ax, strength, [None if record is None else record["ci"] for record in post_records], "Y_post recoverability", "^")
    _errorbar(ax, strength, [None if record is None else record["ci"] for record in pre_records], "Y_pre full-trace recoverability", "D")
    ax.set(xlabel="Verification strength (axial force)", ylabel="Rate / balanced accuracy", ylim=(-0.03, 1.03))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(figures / "gate0_tradeoff.png", dpi=180)
    plt.savefig(figures / "gate0_tradeoff.pdf")
    plt.close()

    fig, axes = plt.subplots(1, len(branch_ids), figsize=(14, 3.2), sharex=True, sharey=True)
    for ax, branch in zip(axes, branch_ids):
        rows = branches[branches["branch"] == branch]
        matrix = np.asarray(
            [
                [((rows.Y_pre == 0) & (rows.Y_post == 0)).sum(), ((rows.Y_pre == 0) & (rows.Y_post == 1)).sum()],
                [((rows.Y_pre == 1) & (rows.Y_post == 0)).sum(), ((rows.Y_pre == 1) & (rows.Y_post == 1)).sum()],
            ]
        )
        image = ax.imshow(matrix, vmin=0, vmax=max(1, matrix.max()), cmap="Blues")
        for row in range(2):
            for column in range(2):
                ax.text(column, row, int(matrix[row, column]), ha="center", va="center")
        ax.set(title=branch, xticks=[0, 1], yticks=[0, 1], xlabel="Y_post")
    axes[0].set_ylabel("Y_pre")
    fig.colorbar(image, ax=axes, shrink=0.75)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.20, top=0.85, wspace=0.35)
    plt.savefig(figures / "transition_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(parents["pre_success_margin"], bins=30, color="#4472c4", alpha=0.85)
    plt.axvline(0.0, color="black", linestyle="--", label="success boundary")
    plt.xlabel("Normalized pre-success margin")
    plt.ylabel("Parent terminal states")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "y_pre_margin_distribution.png", dpi=180)
    plt.close()

    natural_df = pd.DataFrame(naturalness_table)
    probe_ids = ["V2", "V3", "V4"]
    labels = list(config["analysis"]["margin_labels"][:-1])
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(10, 5))
    for index, branch in enumerate(probe_ids):
        subset = natural_df[(natural_df.branch == branch) & natural_df.margin_bin.isin(labels)].set_index("margin_bin")
        values = [subset.loc[label, "vic"] if label in subset.index and pd.notna(subset.loc[label, "vic"]) else 0.0 for label in labels]
        plt.bar(x + (index - 1) * width, values, width=width, label=branch)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("VIC among pre-failures")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "vic_by_pre_success_margin.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for target, marker in (("Y_pre", "o"), ("Y_post", "s")):
        values = [_value(decision, branch, target, "R3_full_trace") for branch in branch_ids]
        plt.plot(strength, values, marker=marker, label=f"{target} full-trace balanced accuracy")
    plt.xlabel("Verification strength (axial force)")
    plt.ylabel("Balanced accuracy")
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "pre_vs_post_recoverability.png", dpi=180)
    plt.close()


def _fmt(value: Any) -> str:
    return "N/A" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.3f}" if isinstance(value, float) else str(value)


def _reports(
    run_dir: Path,
    status: str,
    hypotheses: dict[str, bool],
    reasons: list[str],
    sanity: dict[str, Any],
    parents: pd.DataFrame,
    branch_metrics: dict[str, Any],
    bootstrap: dict[str, Any],
    recovery_records: list[dict[str, Any]],
    recovery: dict[str, Any],
    naturalness: dict[str, Any],
    config: dict[str, Any],
) -> None:
    report_dir = run_dir / "reports"
    falsification_max = recovery["falsification_max"]
    metric_lines = [
        "| Branch | Strength | 0→0 | 0→1 | 1→0 | 1→1 | VIC | VIC 95% CI | VID | Causal soundness | Δcredit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    recovery_lines = ["| Branch | Y_pre R3 suite max (model) | Y_post R3 suite max (model) |", "|---|---:|---:|"]
    near_lines = ["| Branch | Near n | Near VIC | 95% CI |", "|---|---:|---:|---:|"]
    for branch in config["verification"]["branches"]:
        metric = branch_metrics[branch]
        vic_ci = bootstrap[branch]["vic"]
        transitions = metric["transition_counts"]
        metric_lines.append(
            f"| {branch} | {_fmt(metric['verification_strength'])} | {transitions['0 -> 0']} | {transitions['0 -> 1']} | {transitions['1 -> 0']} | {transitions['1 -> 1']} | {_fmt(metric['vic'])} | [{_fmt(vic_ci['low'])}, {_fmt(vic_ci['high'])}] | {_fmt(metric['vid'])} | {_fmt(metric['causal_soundness'])} | {_fmt(metric['delta_credit'])} |"
        )
        recovery_lines.append(
            f"| {branch} | {_fmt(_value(falsification_max, branch, 'Y_pre', 'R3_full_trace'))} ({falsification_max[branch]['Y_pre']['R3_full_trace']['model'] if falsification_max[branch]['Y_pre']['R3_full_trace'] else 'N/A'}) | {_fmt(_value(falsification_max, branch, 'Y_post', 'R3_full_trace'))} ({falsification_max[branch]['Y_post']['R3_full_trace']['model'] if falsification_max[branch]['Y_post']['R3_full_trace'] else 'N/A'}) |"
        )
        near_ci = bootstrap[branch]["vic_near_success"]
        near_lines.append(
            f"| {branch} | {near_ci['n']} | {_fmt(near_ci['estimate'])} | [{_fmt(near_ci['low'])}, {_fmt(near_ci['high'])}] |"
        )
    detailed_recovery_lines = [
        "| Branch | Target | Regime | Model | Train 0/1 | Test 0/1 | Status | Balanced accuracy | 95% CI | ROC-AUC |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for record in recovery_records:
        ci = record.get("ci")
        interval = "N/A" if not ci else f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}]"
        detailed_recovery_lines.append(
            f"| {record['branch']} | {record['target']} | {record['regime']} | {record['model']} | {record['train_negative_n']}/{record['train_positive_n']} | {record['test_negative_n']}/{record['test_positive_n']} | {record['status']} | {_fmt(record['balanced_accuracy'])} | {interval} | {_fmt(record['roc_auc'])} |"
        )
    categories = parents["terminal_category"].value_counts().to_dict()
    prohibited = [
        '"verification erases history"',
        '"pre-verification outcome is information-theoretically unidentifiable"',
        '"this is a widespread robotics problem"',
        '"the effect generalizes beyond this connector model"',
    ]
    if status == "GO":
        next_step = "Propose, but do not implement without new authorization, Gate-1 replication on latch/closure and fastening/seating mechanisms."
    elif status == "CONTINUE":
        next_step = "No Gate-1 is authorized; the only defensible next study is a separately approved causal-miscredit replication without history-erasure claims."
    else:
        next_step = "No further research implementation is authorized under this direction; Gate-0 terminates here."
    report = f"""# Gate-0 Result

## 1. Terminal Decision
{status}

## 2. Exact Scientific Question
Did the connector insertion succeed before verification, or because the verification action caused it to succeed?

## 3. Experimental Setup
CPU-only MuJoCo connector insertion with {len(parents)} natural parent terminal states and five paired branches restored from identical `mjSTATE_INTEGRATION` snapshots.

## 4. Frozen Success Predicate
`depth >= {config['success']['depth_threshold']}`, `lateral <= {config['success']['lateral_threshold']}`, and `tilt <= {config['success']['angular_threshold_deg']}°`. The normalized success margin is the minimum normalized component margin.

## 5. Natural Rollout Distribution
Pre-verification success rate: {_fmt(float(parents['Y_pre'].mean()))}. Categories: `{json.dumps(categories, sort_keys=True)}`.

## 6. Verification Branches
V0 hold, V1 passive read-only, and V2/V3/V4 frozen axial probes. Check A={sanity['check_A_y_pre_identical']}; Check B={sanity['check_B_initial_hash_identical']}; physical distinctness={sanity['check_D_pass']}.

## 7. Verification-Induced Completion
{chr(10).join(metric_lines)}

VIC uses all `Y_pre=0` parents as its frozen denominator; the event additionally requires `control_Y_post=0` and probe `Y_post=1`.

{chr(10).join(near_lines)}

## 8. Verification-Induced Disruption
Reported in the same complete transition metrics table and in `figures/transition_matrix.png`.

## 9. Causal Soundness
The table reports `P(Y_pre=1 | Y_post=1)` for every branch. V0 spontaneous completion rate was {_fmt(sanity['check_C_v0_completion_rate'])}.

## 10. Y_post Recoverability
{chr(10).join(recovery_lines)}

## 11. Y_pre Retrospective Recoverability
R1 final state, R2 trajectory summaries, and R3 resampled full traces were each evaluated with LogisticRegression, RandomForestClassifier, and MLPClassifier using a disjoint parent-level split. The gate's predeclared strongest-baseline rule is `{recovery['falsification_rule']}` across the complete frozen suite. No post-hoc model family is added or removed.

## 12. Full-Trajectory Challenge
The table shows R3 balanced accuracy, not an impossibility bound. Both outcomes use the predeclared suite maximum. Train/test parent overlap: `{recovery['split']['overlap']}`. Per-parent predictions and majority baselines are saved in the results ledgers.

{chr(10).join(detailed_recovery_lines)}

## 13. Success-Margin / Boundary Analysis
Strong-probe completions={naturalness['strong_completion_count']}; very-near fraction={_fmt(naturalness['very_near_fraction'])}; non-very-near completions={naturalness['non_very_near_completion_count']}.

## 14. Main Tradeoff Figure
![Gate-0 tradeoff](../figures/gate0_tradeoff.png)

## 15. Hypothesis Results
""" + "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in hypotheses.items()) + f"""

## 16. Strongest Alternative Explanation
The effect may be a threshold-sensitive artifact of a scripted controller and a primitive square connector, while full contact trajectories may retain enough signatures to reconstruct prior status. The naturalness audit and R3 baselines bound, but do not eliminate, this explanation.

## 17. Gate Decision Rationale
""" + "\n".join(f"- {reason}" for reason in reasons) + f"""

## 18. What Must NOT Be Claimed
""" + "\n".join(f"- DO NOT CLAIM: {claim}" for claim in prohibited) + f"""

## 19. Next Authorized Step
{next_step}
"""
    (report_dir / "gate0_report.md").write_text(report, encoding="utf-8")
    termination = "PASSED_GATE_0" if status == "GO" else "GATE_0"
    decision = f"""# Gate-0 Decision

- `scientific_status = {status}`
- `termination_gate = {termination}`

## Frozen gate rationale
""" + "\n".join(f"- {reason}" for reason in reasons) + f"""

## Hypotheses
""" + "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in hypotheses.items()) + f"""

## Next authorized action
{next_step}
"""
    (report_dir / "gate0_decision.md").write_text(decision, encoding="utf-8")


def analyze_run(run_path: str | Path) -> dict[str, Any]:
    run_dir = Path(run_path).resolve()
    config = load_config(run_dir / "config.yaml")
    manifest = _verify_run_integrity(run_dir, config)
    analysis_provenance = _analysis_provenance(run_dir)
    results = run_dir / "results"
    parents = pd.read_parquet(results / "parent_rollouts.parquet")
    branches = annotate_transitions(pd.read_parquet(results / "verification_branches.parquet"))
    branch_metrics, bootstrap = _branch_metrics(branches, config)
    features = _feature_sets(branches, config, run_dir)
    recovery_records, recovery, predictions, assignments = _recoverability(branches, features, config)
    pd.DataFrame(recovery_records).to_parquet(results / "recoverability_results.parquet", index=False)
    pd.DataFrame(predictions).to_parquet(results / "recoverability_predictions.parquet", index=False)
    pd.DataFrame(assignments).to_parquet(results / "split_assignments.parquet", index=False)
    naturalness_table, naturalness = _naturalness(branches, config)
    pd.DataFrame(naturalness_table).to_parquet(results / "vic_by_margin.parquet", index=False)
    status, hypotheses, sanity, reasons = _decide(branches, branch_metrics, recovery, naturalness, config)
    metrics = {
        "scientific_status": status,
        "termination_gate": "PASSED_GATE_0" if status == "GO" else "GATE_0",
        "hypotheses": hypotheses,
        "branch_metrics": branch_metrics,
        "recoverability": recovery,
        "naturalness": naturalness,
        "sanity_checks": sanity,
        "decision_reasons": reasons,
        "analysis_provenance": analysis_provenance,
    }
    _write_json(results / "metrics.json", metrics)
    _write_json(results / "bootstrap_results.json", bootstrap)
    _figures(run_dir, parents, branches, branch_metrics, bootstrap, recovery, naturalness_table, config)
    _reports(
        run_dir, status, hypotheses, reasons, sanity, parents, branch_metrics, bootstrap,
        recovery_records, recovery, naturalness, config,
    )
    manifest_path = run_dir / "manifest.json"
    analysis_history = manifest.setdefault("analysis_history", [])
    if analysis_provenance not in analysis_history:
        analysis_history.append(analysis_provenance)
    manifest.update(
        scientific_status=status,
        termination_gate=metrics["termination_gate"],
        latest_analysis_provenance=analysis_provenance,
    )
    _write_json(manifest_path, manifest)
    print(f"SCIENTIFIC_STATUS={status}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing run directory")
    args = parser.parse_args()
    analyze_run(args.run)


if __name__ == "__main__":
    main()
