"""MuJoCo simulation primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import mujoco
import numpy as np


def evaluate_success(
    insertion_depth: float,
    lateral_error: float,
    angular_error_deg: float,
    success_config: dict[str, Any],
) -> dict[str, float | bool]:
    depth_margin = insertion_depth - float(success_config["depth_threshold"])
    lateral_margin = float(success_config["lateral_threshold"]) - lateral_error
    angular_margin = float(success_config["angular_threshold_deg"]) - angular_error_deg
    normalized = (
        depth_margin / float(success_config["depth_threshold"]),
        lateral_margin / float(success_config["lateral_threshold"]),
        angular_margin / float(success_config["angular_threshold_deg"]),
    )
    return {
        "success": bool(depth_margin >= 0.0 and lateral_margin >= 0.0 and angular_margin >= 0.0),
        "depth_margin": depth_margin,
        "lateral_margin": lateral_margin,
        "angular_margin": angular_margin,
        "success_margin": min(normalized),
    }


@dataclass(frozen=True)
class TaskResult:
    trajectory: dict[str, np.ndarray]
    snapshot: np.ndarray
    state_hash: str
    terminal_state_id: str
    y_pre: bool
    metrics: dict[str, float | bool]
    initial_conditions: dict[str, Any]
    environment_parameters: dict[str, Any]
    controller_parameters: dict[str, Any]


@dataclass(frozen=True)
class BranchResult:
    parent_episode_id: str
    branch_id: str
    verification_type: str
    verification_strength: float
    initial_state_hash: str
    y_pre: bool
    y_post: bool
    pre_metrics: dict[str, float | bool]
    post_metrics: dict[str, float | bool]
    trajectory: dict[str, np.ndarray]


def _uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _quat_from_tilt(rx_deg: float, ry_deg: float) -> np.ndarray:
    rx = math.radians(rx_deg) / 2.0
    ry = math.radians(ry_deg) / 2.0
    cx, sx, cy, sy = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry)
    return np.asarray([cy * cx, cy * sx, sy * cx, -sy * sx], dtype=np.float64)


class ConnectorEpisode:
    """One naturally randomized task rollout and its paired verification branches."""

    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION

    def __init__(self, config: dict[str, Any], seed: int):
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.params = self._sample_parameters()
        self.model = mujoco.MjModel.from_xml_string(self._model_xml())
        self.model.dof_damping[:3] = float(self.config["model"]["dof_damping"])
        self.model.dof_damping[3:] = float(self.config["model"]["angular_damping"])
        self.data = mujoco.MjData(self.model)
        self.peg_body_id = self.model.body("peg").id
        initial = self.params["initial_conditions"]
        self.data.qpos[:3] = [initial["x"], initial["y"], initial["z"]]
        self.data.qpos[3:7] = _quat_from_tilt(initial["tilt_x_deg"], initial["tilt_y_deg"])
        mujoco.mj_forward(self.model, self.data)

    def _sample_parameters(self) -> dict[str, Any]:
        randomization = self.config["randomization"]
        simulation = self.config["simulation"]
        model = self.config["model"]
        initial = {
            "x": _uniform(self.rng, randomization["initial_xy"]),
            "y": _uniform(self.rng, randomization["initial_xy"]),
            "z": _uniform(self.rng, randomization["initial_z"]),
            "tilt_x_deg": _uniform(self.rng, randomization["initial_tilt_deg"]),
            "tilt_y_deg": _uniform(self.rng, randomization["initial_tilt_deg"]),
        }
        environment = {
            "clearance": _uniform(self.rng, model["clearance"]),
            "friction": _uniform(self.rng, model["friction"]),
        }
        controller = {
            "target_x": _uniform(self.rng, randomization["target_xy_noise"]),
            "target_y": _uniform(self.rng, randomization["target_xy_noise"]),
            "target_depth": float(randomization["nominal_target_depth"])
            + _uniform(self.rng, randomization["target_depth_noise"]),
            "lateral_kp": _uniform(self.rng, randomization["lateral_kp"]),
            "axial_kp": _uniform(self.rng, randomization["axial_kp"]),
            "angular_kp": _uniform(self.rng, randomization["angular_kp"]),
            "controller_noise": _uniform(self.rng, randomization["controller_noise"]),
            "actuation_noise": _uniform(self.rng, randomization["actuation_noise"]),
            "max_force": _uniform(self.rng, randomization["max_force"]),
            "task_steps": int(self.rng.integers(simulation["task_steps"][0], simulation["task_steps"][1] + 1)),
            "settle_steps": int(
                self.rng.integers(simulation["settle_steps"][0], simulation["settle_steps"][1] + 1)
            ),
        }
        return {
            "initial_conditions": initial,
            "environment_parameters": environment,
            "controller_parameters": controller,
        }

    def _model_xml(self) -> str:
        model = self.config["model"]
        half_x, half_y, half_z = (float(v) for v in model["peg_half_size"])
        top = float(model["hole_top_z"])
        depth = float(model["socket_depth"])
        wall = float(model["wall_thickness"])
        inner = max(half_x, half_y) + self.params["environment_parameters"]["clearance"]
        friction = self.params["environment_parameters"]["friction"]
        wall_z = top - depth / 2.0
        floor_z = top - depth - 0.002
        outer = inner + wall
        timestep = float(self.config["simulation"]["timestep"])
        return f"""
<mujoco model="gate0_connector">
  <compiler angle="degree"/>
  <option timestep="{timestep}" gravity="0 0 0" integrator="Euler" iterations="50" cone="elliptic"/>
  <default>
    <geom friction="{friction} 0.005 0.0001" condim="4" solref="0.004 1" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    <body name="socket">
      <geom name="wall_left" type="box" pos="{-inner - wall / 2.0} 0 {wall_z}" size="{wall / 2.0} {outer} {depth / 2.0}"/>
      <geom name="wall_right" type="box" pos="{inner + wall / 2.0} 0 {wall_z}" size="{wall / 2.0} {outer} {depth / 2.0}"/>
      <geom name="wall_front" type="box" pos="0 {-inner - wall / 2.0} {wall_z}" size="{inner} {wall / 2.0} {depth / 2.0}"/>
      <geom name="wall_back" type="box" pos="0 {inner + wall / 2.0} {wall_z}" size="{inner} {wall / 2.0} {depth / 2.0}"/>
      <geom name="socket_floor" type="box" pos="0 0 {floor_z}" size="{outer} {outer} 0.002"/>
    </body>
    <body name="peg" pos="0 0 0">
      <freejoint name="peg_free"/>
      <geom name="peg_geom" type="box" size="{half_x} {half_y} {half_z}" mass="{float(model['peg_mass'])}"/>
    </body>
  </worldbody>
</mujoco>
"""

    def state_vector(self) -> np.ndarray:
        state = np.empty(mujoco.mj_stateSize(self.model, self.state_spec), dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, self.state_spec)
        return state

    def state_hash(self) -> str:
        return hashlib.sha256(self.state_vector().tobytes()).hexdigest()

    def restore(self, snapshot: np.ndarray) -> None:
        mujoco.mj_setState(self.model, self.data, np.asarray(snapshot, dtype=np.float64), self.state_spec)

    def _state_metrics(self) -> dict[str, float | bool]:
        half_z = float(self.config["model"]["peg_half_size"][2])
        peg_pos = self.data.qpos[:3]
        body_z = self.data.xmat[self.peg_body_id].reshape(3, 3)[:, 2]
        insertion_depth = float(self.config["model"]["hole_top_z"]) - float(peg_pos[2] - half_z)
        lateral_error = float(np.linalg.norm(peg_pos[:2]))
        angular_error = math.degrees(math.acos(float(np.clip(body_z[2], -1.0, 1.0))))
        result = evaluate_success(insertion_depth, lateral_error, angular_error, self.config["success"])
        result.update(
            insertion_depth=insertion_depth,
            lateral_error=lateral_error,
            angular_error_deg=angular_error,
        )
        return result

    def _contact_features(self) -> tuple[int, np.ndarray, float]:
        wrench = np.zeros(6, dtype=np.float64)
        max_penetration = 0.0
        for index in range(self.data.ncon):
            contact_wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, index, contact_wrench)
            wrench += contact_wrench
            max_penetration = max(max_penetration, -float(self.data.contact[index].dist))
        return int(self.data.ncon), wrench, max_penetration

    def _record(self, command: np.ndarray, verification_action: float) -> dict[str, Any]:
        metrics = self._state_metrics()
        contact_count, wrench, penetration = self._contact_features()
        peg_pose = self.data.qpos[:7].copy()
        hole_pose = np.asarray(
            [0.0, 0.0, float(self.config["model"]["hole_top_z"]), 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        relative_pose = peg_pose.copy()
        relative_pose[:3] -= hole_pose[:3]
        return {
            "time": float(self.data.time),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "peg_pose": peg_pose,
            "hole_pose": hole_pose,
            "relative_pose": relative_pose,
            "contact_count": contact_count,
            "contact_force": float(np.linalg.norm(wrench[:3])),
            "force_torque": wrench,
            "contact_penetration": penetration,
            "actuator_command": command.copy(),
            "verification_action": float(verification_action),
            "insertion_depth": float(metrics["insertion_depth"]),
            "lateral_error": float(metrics["lateral_error"]),
            "angular_error_deg": float(metrics["angular_error_deg"]),
            "success": bool(metrics["success"]),
        }

    @staticmethod
    def _stack(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        return {key: np.asarray([record[key] for record in records]) for key in records[0]}

    def _task_command(self) -> np.ndarray:
        controller = self.params["controller_parameters"]
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        target_z = (
            float(self.config["model"]["hole_top_z"])
            + float(self.config["model"]["peg_half_size"][2])
            - controller["target_depth"]
        )
        target = np.asarray([controller["target_x"], controller["target_y"], target_z])
        force = np.empty(3, dtype=np.float64)
        force[:2] = controller["lateral_kp"] * (target[:2] - pos[:2]) - 0.8 * vel[:2]
        force[2] = controller["axial_kp"] * (target[2] - pos[2]) - 0.8 * vel[2]
        force += self.rng.normal(0.0, controller["controller_noise"], size=3)
        force += self.rng.normal(0.0, controller["actuation_noise"], size=3)
        force = np.clip(force, -controller["max_force"], controller["max_force"])
        body_z = self.data.xmat[self.peg_body_id].reshape(3, 3)[:, 2]
        torque = controller["angular_kp"] * np.cross(body_z, np.asarray([0.0, 0.0, 1.0]))
        torque -= 0.02 * self.data.qvel[3:6]
        return np.concatenate([force, torque])

    def _step(self, command: np.ndarray) -> None:
        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self.peg_body_id] = command
        mujoco.mj_step(self.model, self.data)

    def run_task(self) -> TaskResult:
        records: list[dict[str, Any]] = []
        controller = self.params["controller_parameters"]
        for _ in range(controller["task_steps"]):
            command = self._task_command()
            self._step(command)
            records.append(self._record(command, 0.0))
        zero = np.zeros(6, dtype=np.float64)
        for _ in range(controller["settle_steps"]):
            self._step(zero)
            records.append(self._record(zero, 0.0))
        snapshot = self.state_vector()
        state_hash = hashlib.sha256(snapshot.tobytes()).hexdigest()
        metrics = self._state_metrics()
        return TaskResult(
            trajectory=self._stack(records),
            snapshot=snapshot,
            state_hash=state_hash,
            terminal_state_id=state_hash[:16],
            y_pre=bool(metrics["success"]),
            metrics=metrics,
            initial_conditions=self.params["initial_conditions"],
            environment_parameters=self.params["environment_parameters"],
            controller_parameters=self.params["controller_parameters"],
        )

    def run_branch(self, task: TaskResult, parent_episode_id: str, branch_id: str) -> BranchResult:
        branch_config = self.config["verification"]["branches"][branch_id]
        self.restore(task.snapshot)
        initial_hash = self.state_hash()
        if initial_hash != task.state_hash:
            raise RuntimeError("restored branch state does not match parent snapshot")
        mujoco.mj_forward(self.model, self.data)
        strength = float(branch_config["strength"])
        command = np.zeros(6, dtype=np.float64)
        if branch_config["type"] == "axial_probe":
            command[2] = -strength
        records: list[dict[str, Any]] = []
        for _ in range(int(self.config["simulation"]["verification_steps"])):
            self._step(command)
            records.append(self._record(command, strength))
        post_metrics = self._state_metrics()
        return BranchResult(
            parent_episode_id=parent_episode_id,
            branch_id=branch_id,
            verification_type=str(branch_config["type"]),
            verification_strength=strength,
            initial_state_hash=initial_hash,
            y_pre=task.y_pre,
            y_post=bool(post_metrics["success"]),
            pre_metrics=task.metrics,
            post_metrics=post_metrics,
            trajectory=self._stack(records),
        )

    def run_all_branches(self, task: TaskResult, parent_episode_id: str) -> list[BranchResult]:
        return [
            self.run_branch(task, parent_episode_id=parent_episode_id, branch_id=branch_id)
            for branch_id in self.config["verification"]["branches"]
        ]
