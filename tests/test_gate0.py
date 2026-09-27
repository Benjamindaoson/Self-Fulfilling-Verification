from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import gate0.analyze as analyze_module
import gate0.config as config_module
import gate0.metrics as metrics_module
import gate0.run as run_module
import gate0.sim as sim_module
import numpy as np
import pandas as pd
import pytest


SUCCESS = {
    "depth_threshold": 0.028,
    "lateral_threshold": 0.0015,
    "angular_threshold_deg": 3.0,
}


def test_success_predicate_requires_every_frozen_component() -> None:
    evaluate = getattr(sim_module, "evaluate_success", None)
    assert callable(evaluate), "evaluate_success must implement the frozen predicate"

    passed = evaluate(0.030, 0.0010, 2.0, SUCCESS)
    failed_depth = evaluate(0.027, 0.0010, 2.0, SUCCESS)
    failed_lateral = evaluate(0.030, 0.0020, 2.0, SUCCESS)
    failed_angle = evaluate(0.030, 0.0010, 4.0, SUCCESS)

    assert passed["success"] is True
    assert passed["depth_margin"] == pytest.approx(0.002)
    assert failed_depth["success"] is False
    assert failed_lateral["success"] is False
    assert failed_angle["success"] is False
    assert failed_lateral["success_margin"] < 0.0


def test_manifest_hash_is_stable_under_mapping_order() -> None:
    config_hash = getattr(config_module, "config_hash", None)
    assert callable(config_hash), "config_hash must canonicalize mappings"

    left = {"experiment": {"seed": 7, "name": "gate0"}, "branches": ["V0", "V2"]}
    right = {"branches": ["V0", "V2"], "experiment": {"name": "gate0", "seed": 7}}
    changed = copy.deepcopy(right)
    changed["experiment"]["seed"] = 8

    assert config_hash(left) == config_hash(right)
    assert config_hash(left) != config_hash(changed)


def _short_config() -> dict:
    config = config_module.load_config(Path("configs/gate0_smoke.yaml"))
    config["simulation"]["task_steps"] = [80, 80]
    config["simulation"]["settle_steps"] = [30, 30]
    config["simulation"]["verification_steps"] = 16
    return config


def test_state_clone_identity_restores_complete_integration_state() -> None:
    episode_type = getattr(sim_module, "ConnectorEpisode", None)
    assert episode_type is not None, "ConnectorEpisode must own real MuJoCo state"
    episode = episode_type(_short_config(), seed=123)
    task = episode.run_task()
    expected = task.snapshot.copy()

    episode.data.qpos[0] += 0.01
    episode.restore(task.snapshot)

    np.testing.assert_array_equal(episode.state_vector(), expected)
    assert episode.state_hash() == task.state_hash


def test_clone_branch_reproducibility_for_deterministic_v0() -> None:
    episode_type = getattr(sim_module, "ConnectorEpisode", None)
    assert episode_type is not None
    episode = episode_type(_short_config(), seed=124)
    task = episode.run_task()

    first = episode.run_branch(task, parent_episode_id="p-124", branch_id="V0")
    second = episode.run_branch(task, parent_episode_id="p-124", branch_id="V0")

    assert first.y_post == second.y_post
    for key in first.trajectory:
        np.testing.assert_array_equal(first.trajectory[key], second.trajectory[key])


def test_y_pre_is_immutable_after_branch_start() -> None:
    episode_type = getattr(sim_module, "ConnectorEpisode", None)
    assert episode_type is not None
    episode = episode_type(_short_config(), seed=125)
    task = episode.run_task()
    original_snapshot = task.snapshot.copy()
    original_y_pre = task.y_pre

    branch = episode.run_branch(task, parent_episode_id="p-125", branch_id="V4")

    assert branch.y_pre == original_y_pre
    assert task.y_pre == original_y_pre
    np.testing.assert_array_equal(task.snapshot, original_snapshot)


def test_v0_no_action_behavior_records_zero_command() -> None:
    episode_type = getattr(sim_module, "ConnectorEpisode", None)
    assert episode_type is not None
    episode = episode_type(_short_config(), seed=126)
    task = episode.run_task()

    branch = episode.run_branch(task, parent_episode_id="p-126", branch_id="V0")

    np.testing.assert_array_equal(branch.trajectory["actuator_command"], 0.0)


def test_branch_trace_starts_before_verification_action() -> None:
    episode = sim_module.ConnectorEpisode(_short_config(), seed=129)
    task = episode.run_task()
    episode.restore(task.snapshot)
    expected_qpos = episode.data.qpos.copy()

    branch = episode.run_branch(task, parent_episode_id="p-129", branch_id="V4")

    np.testing.assert_array_equal(branch.trajectory["qpos"][0], expected_qpos)
    np.testing.assert_array_equal(branch.trajectory["actuator_command"][0], 0.0)
    assert branch.trajectory["verification_action"][0] == 0.0


def test_contact_parameters_are_randomized_before_task_and_recorded() -> None:
    config = _short_config()
    episode = sim_module.ConnectorEpisode(config, seed=130)
    environment = episode.params["environment_parameters"]

    for key in ("contact_timeconst", "contact_dampratio", "contact_width"):
        low, high = config["model"][key]
        assert low <= environment[key] <= high


def test_branch_parent_linkage_and_initial_state_identity() -> None:
    episode_type = getattr(sim_module, "ConnectorEpisode", None)
    assert episode_type is not None
    episode = episode_type(_short_config(), seed=127)
    task = episode.run_task()

    branches = episode.run_all_branches(task, parent_episode_id="p-127")

    assert {branch.branch_id for branch in branches} == {"V0", "V1", "V2", "V3", "V4"}
    assert {branch.parent_episode_id for branch in branches} == {"p-127"}
    assert {branch.initial_state_hash for branch in branches} == {task.state_hash}
    assert {branch.y_pre for branch in branches} == {task.y_pre}


def _metric_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"parent_episode_id": "p0", "branch": "V0", "Y_pre": 0, "Y_post": 0},
            {"parent_episode_id": "p0", "branch": "V2", "Y_pre": 0, "Y_post": 1},
            {"parent_episode_id": "p1", "branch": "V0", "Y_pre": 0, "Y_post": 0},
            {"parent_episode_id": "p1", "branch": "V2", "Y_pre": 0, "Y_post": 0},
            {"parent_episode_id": "p2", "branch": "V0", "Y_pre": 1, "Y_post": 1},
            {"parent_episode_id": "p2", "branch": "V2", "Y_pre": 1, "Y_post": 0},
            {"parent_episode_id": "p3", "branch": "V0", "Y_pre": 1, "Y_post": 1},
            {"parent_episode_id": "p3", "branch": "V2", "Y_pre": 1, "Y_post": 1},
        ]
    )


def test_metric_vic_uses_paired_control_and_pre_failure_denominator() -> None:
    annotate = getattr(metrics_module, "annotate_transitions", None)
    summarize = getattr(metrics_module, "summarize_branch", None)
    assert callable(annotate) and callable(summarize)

    annotated = annotate(_metric_fixture())
    summary = summarize(annotated[annotated["branch"] == "V2"])

    assert annotated.loc[(annotated.parent_episode_id == "p0") & (annotated.branch == "V2"), "verification_induced_completion"].item()
    assert summary["vic"] == pytest.approx(0.5)
    assert summary["transition_counts"] == {"0 -> 0": 1, "0 -> 1": 1, "1 -> 0": 1, "1 -> 1": 1}


def test_metric_vid_counts_success_destroyed_by_verification() -> None:
    annotate = getattr(metrics_module, "annotate_transitions", None)
    summarize = getattr(metrics_module, "summarize_branch", None)
    assert callable(annotate) and callable(summarize)

    annotated = annotate(_metric_fixture())
    summary = summarize(annotated[annotated["branch"] == "V2"])

    assert annotated.loc[(annotated.parent_episode_id == "p2") & (annotated.branch == "V2"), "verification_induced_disruption"].item()
    assert summary["vid"] == pytest.approx(0.5)
    assert summary["causal_soundness"] == pytest.approx(0.5)
    assert summary["delta_credit"] == pytest.approx(0.0)


def test_group_split_has_no_parent_leakage() -> None:
    group_split = getattr(metrics_module, "group_split", None)
    assert callable(group_split)
    groups = np.asarray(["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"])

    train, test = group_split(groups, test_fraction=0.25, seed=9)

    assert set(groups[train]).isdisjoint(set(groups[test]))
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(len(groups)))


def test_force_impulse_uses_seconds_not_raw_force_sum() -> None:
    episode = sim_module.ConnectorEpisode(_short_config(), seed=128)
    task = episode.run_task()
    branch = episode.run_branch(task, parent_episode_id="p-128", branch_id="V2")
    branch.trajectory["contact_force"] = np.asarray([2.0, 3.0])
    branch.trajectory["contact_count"] = np.asarray([1, 1])
    trace = {"trajectory_path": "trace.npz", "sha256": "abc", "timesteps": 2}

    row = run_module._branch_row(128, branch, trace, timestep=0.002)

    assert row["force_impulse"] == pytest.approx(0.006)


def test_config_rejects_unknown_verification_type(tmp_path: Path) -> None:
    config = config_module.load_config(Path("configs/gate0_smoke.yaml"))
    config["verification"]["branches"]["V3"]["type"] = "axial_proeb"
    path = tmp_path / "invalid.yaml"
    path.write_text(__import__("yaml").safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="verification.*V3.*type"):
        config_module.load_config(path)


def test_config_rejects_fractional_counts_and_missing_gate(tmp_path: Path) -> None:
    config = config_module.load_config(Path("configs/gate0_smoke.yaml"))
    config["experiment"]["parent_count"] = 20.5
    path = tmp_path / "fractional.yaml"
    path.write_text(__import__("yaml").safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="parent_count.*integer"):
        config_module.load_config(path)

    config["experiment"]["parent_count"] = 20
    config["gate"].pop("max_v0_completion")
    path.write_text(__import__("yaml").safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="gate.*max_v0_completion"):
        config_module.load_config(path)


def test_run_rejects_path_escaping_run_id_before_writing() -> None:
    with pytest.raises(ValueError, match="run_id"):
        run_module.run_experiment(Path("configs/gate0_smoke.yaml"), run_id="../escape")


def test_suite_max_near_perfect_y_pre_recovery_unconditionally_kills() -> None:
    config = config_module.load_config(Path("configs/gate0_smoke.yaml"))
    branches = pd.DataFrame([
        {"parent_episode_id": "p0", "branch": "V0", "Y_pre": 0, "Y_post": 0, "initial_state_hash": "same", "pre_insertion_depth": 0.0, "post_insertion_depth": 0.0},
        {"parent_episode_id": "p0", "branch": "V2", "Y_pre": 0, "Y_post": 1, "initial_state_hash": "same", "pre_insertion_depth": 0.0, "post_insertion_depth": 0.001},
        {"parent_episode_id": "p0", "branch": "V3", "Y_pre": 0, "Y_post": 1, "initial_state_hash": "same", "pre_insertion_depth": 0.0, "post_insertion_depth": 0.002},
        {"parent_episode_id": "p0", "branch": "V4", "Y_pre": 0, "Y_post": 1, "initial_state_hash": "same", "pre_insertion_depth": 0.0, "post_insertion_depth": 0.003},
    ])
    branch_metrics = {
        "V2": {"vic": 0.10, "causal_soundness": 0.90},
        "V3": {"vic": 0.15, "causal_soundness": 0.85},
        "V4": {"vic": 0.20, "causal_soundness": 0.80},
    }
    suite = {
        branch: {
            "Y_pre": {"R3_full_trace": {"balanced_accuracy": 0.99}},
            "Y_post": {"R3_full_trace": {"balanced_accuracy": 0.90}},
        }
        for branch in ("V2", "V3", "V4")
    }
    recovery = {"falsification_max": suite}
    naturalness = {"very_near_fraction": 0.5, "non_very_near_completion_count": 1}

    status, hypotheses, _, _ = analyze_module._decide(
        branches, branch_metrics, recovery, naturalness, config
    )

    assert hypotheses["H4"] is True
    assert hypotheses["H5"] is False
    assert status == "KILL"


def test_run_integrity_rejects_mutated_config_before_analysis(tmp_path: Path) -> None:
    config = config_module.load_config(Path("configs/gate0_smoke.yaml"))
    run_dir = tmp_path / "run"
    (run_dir / "results").mkdir(parents=True)
    config_path = run_dir / "config.yaml"
    config_path.write_text(__import__("yaml").safe_dump(config), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"config_hash": "0" * 64}), encoding="utf-8"
    )

    verify = getattr(analyze_module, "_verify_run_integrity", None)
    assert callable(verify), "analysis must fail closed on changed scientific inputs"
    with pytest.raises(RuntimeError, match="config hash mismatch"):
        verify(run_dir, config)


def test_run_integrity_rejects_mutated_trajectory(tmp_path: Path) -> None:
    config = config_module.load_config(Path("configs/gate0_smoke.yaml"))
    run_dir = tmp_path / "run"
    results = run_dir / "results"
    trajectories = run_dir / "trajectories"
    results.mkdir(parents=True)
    trajectories.mkdir()
    config_path = run_dir / "config.yaml"
    config_path.write_text(__import__("yaml").safe_dump(config), encoding="utf-8")
    trace = trajectories / "trace.npz"
    trace.write_bytes(b"changed")
    pd.DataFrame(
        [{"trajectory_path": "trajectories/trace.npz", "sha256": hashlib.sha256(b"original").hexdigest()}]
    ).to_parquet(results / "trajectory_index.parquet", index=False)
    pd.DataFrame([{"snapshot_path": "snapshots/unused.npy", "snapshot_hash": "unused"}]).to_parquet(
        results / "parent_rollouts.parquet", index=False
    )
    pd.DataFrame([{"branch": "V0"}]).to_parquet(results / "verification_branches.parquet", index=False)
    ledger_hashes = {
        path.relative_to(run_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in results.glob("*.parquet")
    }
    (run_dir / "manifest.json").write_text(
        json.dumps({"config_hash": config_module.config_hash(config), "input_ledger_hashes": ledger_hashes}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="trajectory hash mismatch"):
        analyze_module._verify_run_integrity(run_dir, config)
