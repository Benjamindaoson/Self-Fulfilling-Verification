"""Run frozen natural-rollout and paired-branch Gate-0 experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any

import mujoco
import numpy as np
import pandas as pd

from gate0 import __version__
from gate0.config import config_hash, load_config
from gate0.metrics import annotate_transitions
from gate0.sim import BranchResult, ConnectorEpisode, TaskResult


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _save_trace(path: Path, trajectory: dict[str, np.ndarray], run_dir: Path) -> dict[str, Any]:
    np.savez_compressed(path, **trajectory)
    return {
        "trajectory_path": path.relative_to(run_dir).as_posix(),
        "timesteps": int(len(trajectory["time"])),
        "sha256": _sha256(path),
    }


def _category(task: TaskResult) -> str:
    if task.y_pre:
        return "successful"
    if float(task.metrics["success_margin"]) >= -0.25:
        return "near-success"
    if int(task.trajectory["contact_count"][-1]) > 0 and float(task.metrics["depth_margin"]) < 0:
        return "jammed"
    if float(task.metrics["lateral_margin"]) < 0 or float(task.metrics["angular_margin"]) < 0:
        return "misaligned"
    if float(task.metrics["depth_margin"]) < 0:
        return "under-inserted"
    return "clear-failure"


def _parent_row(parent_id: str, seed: int, task: TaskResult, trace: dict[str, Any], snapshot_path: Path) -> dict[str, Any]:
    return {
        "episode_id": parent_id,
        "seed": seed,
        "initial_conditions": json.dumps(task.initial_conditions, sort_keys=True),
        "environment_parameters": json.dumps(task.environment_parameters, sort_keys=True),
        "controller_parameters": json.dumps(task.controller_parameters, sort_keys=True),
        "task_trajectory_path": trace["trajectory_path"],
        "task_trajectory_sha256": trace["sha256"],
        "task_timesteps": trace["timesteps"],
        "terminal_state_id": task.terminal_state_id,
        "snapshot_path": snapshot_path.as_posix(),
        "snapshot_hash": task.state_hash,
        "Y_pre": int(task.y_pre),
        "insertion_depth": float(task.metrics["insertion_depth"]),
        "lateral_error": float(task.metrics["lateral_error"]),
        "angular_error_deg": float(task.metrics["angular_error_deg"]),
        "depth_margin": float(task.metrics["depth_margin"]),
        "lateral_margin": float(task.metrics["lateral_margin"]),
        "angular_margin": float(task.metrics["angular_margin"]),
        "pre_success_margin": float(task.metrics["success_margin"]),
        "terminal_category": _category(task),
    }


def _branch_row(seed: int, branch: BranchResult, trace: dict[str, Any], timestep: float) -> dict[str, Any]:
    trajectory = branch.trajectory
    return {
        "parent_episode_id": branch.parent_episode_id,
        "seed": seed,
        "branch": branch.branch_id,
        "verification_type": branch.verification_type,
        "verification_strength": branch.verification_strength,
        "initial_state_hash": branch.initial_state_hash,
        "Y_pre": int(branch.y_pre),
        "Y_post": int(branch.y_post),
        "pre_insertion_depth": float(branch.pre_metrics["insertion_depth"]),
        "post_insertion_depth": float(branch.post_metrics["insertion_depth"]),
        "pre_lateral_error": float(branch.pre_metrics["lateral_error"]),
        "post_lateral_error": float(branch.post_metrics["lateral_error"]),
        "pre_angular_error_deg": float(branch.pre_metrics["angular_error_deg"]),
        "post_angular_error_deg": float(branch.post_metrics["angular_error_deg"]),
        "pre_success_margin": float(branch.pre_metrics["success_margin"]),
        "post_success_margin": float(branch.post_metrics["success_margin"]),
        "depth_margin": float(branch.post_metrics["depth_margin"]),
        "lateral_margin": float(branch.post_metrics["lateral_margin"]),
        "angular_margin": float(branch.post_metrics["angular_margin"]),
        "final_contact_count": int(trajectory["contact_count"][-1]),
        "final_contact_force": float(trajectory["contact_force"][-1]),
        "max_contact_force": float(np.max(trajectory["contact_force"])),
        "force_impulse": float(np.sum(trajectory["contact_force"]) * timestep),
        "trajectory_path": trace["trajectory_path"],
        "trajectory_sha256": trace["sha256"],
        "trajectory_timesteps": trace["timesteps"],
    }


def run_experiment(config_path: str | Path, run_id: str | None = None) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    digest = config_hash(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"{timestamp}_{config['experiment']['name']}_{digest[:8]}"
    run_dir = Path(config["experiment"]["output_root"]).resolve() / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    for directory in ("results", "trajectories", "snapshots", "figures", "reports"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    manifest = {
        "run_id": run_id,
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_hash": digest,
        "code_version": __version__,
        "python_version": platform.python_version(),
        "mujoco_version": mujoco.__version__,
        "seed_range": [
            int(config["experiment"]["seed_start"]),
            int(config["experiment"]["seed_start"]) + int(config["experiment"]["parent_count"]) - 1,
        ],
        "parent_count_requested": int(config["experiment"]["parent_count"]),
        "branches": list(config["verification"]["branches"]),
        "success_predicate_config": config["success"],
        "verification_config": config["verification"],
        "environment_randomization_config": {
            "model": config["model"],
            "randomization": config["randomization"],
            "simulation": config["simulation"],
        },
        "gate_config": config["gate"],
    }
    _write_json(run_dir / "manifest.json", manifest)

    parents: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    try:
        total = int(config["experiment"]["parent_count"])
        seed_start = int(config["experiment"]["seed_start"])
        for index in range(total):
            seed = seed_start + index
            parent_id = f"p{index:05d}"
            episode = ConnectorEpisode(config, seed)
            task = episode.run_task()
            task_path = run_dir / "trajectories" / f"{parent_id}_TASK.npz"
            task_trace = _save_trace(task_path, task.trajectory, run_dir)
            snapshot_path = run_dir / "snapshots" / f"{parent_id}.npy"
            np.save(snapshot_path, task.snapshot, allow_pickle=False)
            parents.append(_parent_row(parent_id, seed, task, task_trace, snapshot_path.relative_to(run_dir)))
            trajectory_rows.append(
                {"parent_episode_id": parent_id, "phase": "TASK", "branch": "TASK", **task_trace}
            )
            for branch in episode.run_all_branches(task, parent_episode_id=parent_id):
                branch_path = run_dir / "trajectories" / f"{parent_id}_{branch.branch_id}.npz"
                branch_trace = _save_trace(branch_path, branch.trajectory, run_dir)
                branch_rows.append(_branch_row(seed, branch, branch_trace, float(config["simulation"]["timestep"])))
                trajectory_rows.append(
                    {
                        "parent_episode_id": parent_id,
                        "phase": "VERIFICATION",
                        "branch": branch.branch_id,
                        **branch_trace,
                    }
                )
            if (index + 1) % max(1, min(25, total)) == 0 or index + 1 == total:
                print(f"rollouts {index + 1}/{total}", flush=True)

        parents_df = pd.DataFrame(parents)
        branches_df = annotate_transitions(pd.DataFrame(branch_rows))
        trajectory_df = pd.DataFrame(trajectory_rows)
        results_dir = run_dir / "results"
        parents_df.to_parquet(results_dir / "parent_rollouts.parquet", index=False)
        branches_df.to_parquet(results_dir / "verification_branches.parquet", index=False)
        trajectory_df.to_parquet(results_dir / "trajectory_index.parquet", index=False)
        manifest.update(
            status="SIMULATION_COMPLETE",
            parent_count=int(len(parents_df)),
            branch_count=int(len(branches_df)),
            trajectory_count=int(len(trajectory_df)),
            simulation_completed_at=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(run_dir / "manifest.json", manifest)

        from gate0.analyze import analyze_run

        analyze_run(run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(status="COMPLETE", completed_at=datetime.now(timezone.utc).isoformat())
        _write_json(run_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="FAILED", failed_at=datetime.now(timezone.utc).isoformat(), error=repr(exc))
        _write_json(run_dir / "manifest.json", manifest)
        raise
    print(f"RUN_DIR={run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen YAML configuration")
    parser.add_argument("--run-id", help="Optional explicit immutable run identifier")
    args = parser.parse_args()
    run_experiment(args.config, args.run_id)


if __name__ == "__main__":
    main()
