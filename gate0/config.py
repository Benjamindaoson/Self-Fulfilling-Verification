"""Configuration loading and stable experiment hashing."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{path} must be <= {maximum}")
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if value < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    return value


def safe_component(value: Any, path: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{path} must be one safe path component")
    return value


def _range(value: Any, path: str, *, positive: bool = False, integer: bool = False, nonnegative: bool = False) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{path} must be a two-element list")
    check = _integer if integer else _number
    low, high = (check(item, f"{path}[{index}]") for index, item in enumerate(value))
    if low > high or (positive and low <= 0) or (nonnegative and low < 0):
        raise ValueError(f"{path} must be an ascending{' positive' if positive else ''} range")


def validate_config(config: dict[str, Any]) -> None:
    """Validate the frozen experiment boundary before any run files are created."""
    required = {"experiment", "simulation", "model", "success", "randomization", "verification", "analysis", "gate"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing configuration sections: {', '.join(missing)}")
    experiment = _mapping(config["experiment"], "experiment")
    safe_component(experiment.get("name"), "experiment.name")
    if not isinstance(experiment.get("output_root"), str) or not experiment["output_root"].strip():
        raise ValueError("experiment.output_root must be a non-empty string")
    for key in ("parent_count", "seed_start"):
        _integer(experiment.get(key), f"experiment.{key}", minimum=1 if key == "parent_count" else 0)

    simulation = _mapping(config["simulation"], "simulation")
    _number(simulation.get("timestep"), "simulation.timestep", minimum=1e-12)
    for key in ("task_steps", "settle_steps"):
        _range(simulation.get(key), f"simulation.{key}", positive=True, integer=True)
    _integer(simulation.get("verification_steps"), "simulation.verification_steps", minimum=1)

    model = _mapping(config["model"], "model")
    for key in ("peg_half_size",):
        value = model.get(key)
        if not isinstance(value, list) or len(value) != 3 or any(_number(item, f"model.{key}", minimum=1e-12) <= 0 for item in value):
            raise ValueError(f"model.{key} must contain three positive values")
    for key in ("hole_top_z", "socket_depth", "wall_thickness", "peg_mass", "dof_damping", "angular_damping"):
        _number(model.get(key), f"model.{key}", minimum=0)
    for key in ("clearance", "friction", "contact_timeconst", "contact_dampratio", "contact_width"):
        _range(model.get(key), f"model.{key}", positive=True)

    success = _mapping(config["success"], "success")
    for key in ("depth_threshold", "lateral_threshold", "angular_threshold_deg"):
        _number(success.get(key), f"success.{key}", minimum=1e-12)

    randomization = _mapping(config["randomization"], "randomization")
    for key in (
        "initial_xy", "initial_z", "initial_tilt_deg", "target_xy_noise", "target_depth_noise",
    ):
        _range(randomization.get(key), f"randomization.{key}")
    for key in ("lateral_kp", "axial_kp", "angular_kp", "max_force"):
        _range(randomization.get(key), f"randomization.{key}", positive=True)
    for key in ("controller_noise", "actuation_noise"):
        _range(randomization.get(key), f"randomization.{key}", nonnegative=True)
    _number(randomization.get("nominal_target_depth"), "randomization.nominal_target_depth", minimum=0)

    verification = _mapping(config["verification"], "verification")
    branches = _mapping(verification.get("branches"), "verification.branches")
    if list(branches) != ["V0", "V1", "V2", "V3", "V4"]:
        raise ValueError("verification.branches must be ordered exactly V0,V1,V2,V3,V4")
    expected_types = {"V0": "hold", "V1": "passive", "V2": "axial_probe", "V3": "axial_probe", "V4": "axial_probe"}
    strengths = []
    for branch, expected_type in expected_types.items():
        spec = _mapping(branches[branch], f"verification.branches.{branch}")
        if spec.get("type") != expected_type:
            raise ValueError(f"verification.branches.{branch}.type must be {expected_type}")
        strengths.append(_number(spec.get("strength"), f"verification.branches.{branch}.strength", minimum=0))
    if strengths[:2] != [0.0, 0.0] or not 0 < strengths[2] < strengths[3] < strengths[4]:
        raise ValueError("verification strengths must be 0,0, then strictly increasing positive probes")

    analysis = _mapping(config["analysis"], "analysis")
    _integer(analysis.get("bootstrap_samples"), "analysis.bootstrap_samples", minimum=1)
    _number(analysis.get("confidence_level"), "analysis.confidence_level", minimum=1e-12, maximum=1 - 1e-12)
    _number(analysis.get("test_fraction"), "analysis.test_fraction", minimum=1e-12, maximum=1 - 1e-12)
    for key in ("split_seed", "classifier_seed"):
        _integer(analysis.get(key), f"analysis.{key}", minimum=0)
    _integer(analysis.get("resample_steps"), "analysis.resample_steps", minimum=2)
    if analysis.get("falsification_rule") != "max_predeclared_suite":
        raise ValueError("analysis.falsification_rule must be max_predeclared_suite")
    bins, labels = analysis.get("margin_bins"), analysis.get("margin_labels")
    if not isinstance(bins, list) or not isinstance(labels, list) or len(bins) != len(labels) + 1:
        raise ValueError("analysis.margin_bins must contain one more item than margin_labels")
    if any(float(left) >= float(right) for left, right in zip(bins, bins[1:])):
        raise ValueError("analysis.margin_bins must be strictly increasing")

    gate = _mapping(config["gate"], "gate")
    gate_keys = {
        "min_vic", "min_vic_increase", "min_post_recoverability", "min_pre_recoverability_drop",
        "max_full_trace_pre_recoverability", "max_v0_completion", "max_strong_probe_vic",
        "max_very_near_vic_fraction",
    }
    if set(gate) != gate_keys:
        missing = sorted(gate_keys - gate.keys())
        extra = sorted(gate.keys() - gate_keys)
        raise ValueError(f"gate keys mismatch; missing={missing}, extra={extra}")
    for key, value in gate.items():
        _number(value, f"gate.{key}", minimum=0, maximum=1)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")
    validate_config(loaded)
    return loaded
