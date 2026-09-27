from __future__ import annotations

import copy
from pathlib import Path

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

    assert row["force_impulse"] == pytest.approx(0.010)
