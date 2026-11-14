"""CPU-MuJoCo + SB3 Gymnasium env for the XgoDuck sit->stand task (Mac training).

Faithful-ish port of the mjlab Mjlab-SitStand task to plain CPU MuJoCo so PPO can
train on Apple Silicon (mujoco_warp's CPU backend is a slow debug path). Reuses:
  - the exact 61-D actor obs layout (tasks/symmetry.py),
  - the BAM actuator model via bam.mujoco.MujocoController (raw-MuJoCo torque+friction),
  - the posture_composite goal score + backward_support_margin (whole-body CoM over
    the heel) reward terms from tasks/mdp.py,
  - the +/-22 deg XGODUCK_HOME_FRAME and SIT/STAND keyframes.

Optionally fits the +15 mm heel clip (heel=True) -> robot_allcollisions_heel scene.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from bam.model import load_model as bam_load_model
from bam.mujoco import MujocoController

_ROBOT_DIR = Path(__file__).resolve().parent.parent / "src" / "mjlab_microduck" / "robot" / "xgoduck"
_BAM_JSON = _ROBOT_DIR / "params" / "1910_m6.json"

_DEG22 = math.radians(22.0)
# XGODUCK_HOME_FRAME (xgoduck_constants.py), keyed by joint-name substring.
_HOME = {
    "hip_yaw": 0.0,
    "left_hip_roll": -0.0873, "right_hip_roll": 0.0873,
    "left_hip_pitch": -_DEG22, "right_hip_pitch": _DEG22,
    "left_knee": -0.0049, "right_knee": 0.0049,
    "left_ankle": _DEG22, "right_ankle": -_DEG22,
    "neck_pitch": 0.3491, "head_pitch": 0.3491,
    "head_yaw": 0.0, "head_roll": 0.0,
}
_STAND_Z = 0.139158
_SIT_Z = 0.085868
_NECK = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def _home_for(jname: str) -> float:
    for k, v in _HOME.items():
        if k in jname:
            return v
    return 0.0


class DuckSitStandEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        heel: bool = False,
        decimation: int = 5,
        episode_s: float = 6.0,
        reset_stand_prob: float = 0.25,
        fall_penalty: float = -20.0,
        alive_bonus: float = 0.5,
        heel_offset: float | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.heel = heel
        self.decimation = decimation
        self.reset_stand_prob = reset_stand_prob
        self.fall_penalty = fall_penalty
        self.alive_bonus = alive_bonus
        self.rng = np.random.default_rng(seed)

        scene = "scene_heel.xml" if heel else "scene.xml"
        spec = mujoco.MjSpec.from_file(str(_ROBOT_DIR / scene))

        # --- convert position servos -> motor actuators for the BAM controller ---
        self.joint_names = []
        for j in spec.joints:
            if j.name and not j.name.startswith("passive_") and j.type == mujoco.mjtJoint.mjJNT_HINGE:
                self.joint_names.append(j.name)
        # keep only joints that have an actuator in the XML
        act_targets = {a.target for a in spec.actuators}
        self.joint_names = [n for n in self.joint_names if n in act_targets]
        self.n_act = len(self.joint_names)

        bam_model = bam_load_model(str(_BAM_JSON))
        bam_model.actuator.kp = 5.0  # kp_fw matches xgoduck_constants
        act = bam_model.actuator
        # STS3215 rate-limits the goal via q_target_smooth; seed it (mjlab does
        # this in FrictionDRBamActuator.initialize — load_log never runs here).
        if hasattr(act, "default_max_velocity") and not hasattr(act, "q_target_smooth"):
            act.q_target_smooth = 0.0
        force_limit = 8.0 * bam_model.kt.value / bam_model.R.value
        armature = act.get_extra_inertia()
        for mjact in spec.actuators:
            if mjact.target in self.joint_names:
                mjact.set_to_motor()
                mjact.forcelimited = True
                mjact.forcerange = (-force_limit, force_limit)
                mjact.gear = [1.0, 0, 0, 0, 0, 0]
        for j in spec.joints:
            if j.name in self.joint_names:
                j.armature = float(armature)
                j.damping = np.zeros((3, 1))
                j.frictionloss = 0.0

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.max_steps = int(episode_s / (self.dt * self.decimation))

        # BAM controller (writes torque + updates friction each physics step)
        self.bam_model = bam_model
        self.ctrl = MujocoController(
            bam_model, self.joint_names, self.model, self.data,
            vin_drop_gain=0.1, vin_min=7.0,
        )

        # joint / dof / qpos addresses, home pose, actuator ctrl order
        self.jids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names])
        self.qpos_adr = self.model.jnt_qposadr[self.jids]
        self.dof_adr = self.model.jnt_dofadr[self.jids]
        self.home = np.array([_home_for(n) for n in self.joint_names])
        self.act_idx = np.array([self.model.actuator(n).id for n in self.joint_names])
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.foot_sites = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("left_foot", "right_foot")]
        self.neck_idx = np.array([i for i, n in enumerate(self.joint_names) if any(k in n for k in _NECK)])
        self.leg_idx = np.array([i for i, n in enumerate(self.joint_names) if not any(k in n for k in _NECK)])

        # heel_offset: foot_site_x - heel_x_min (stock 0.0329, heel 0.0479)
        self.heel_offset = heel_offset if heel_offset is not None else (0.0479 if heel else 0.0329)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (61,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.n_act,), dtype=np.float32)
        self._last_action = np.zeros(self.n_act)
        self._prev_trunk_z = _STAND_Z
        self._steps = 0

    # ---------------- obs ----------------
    def _quat_rotate_inv(self, q, v):
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
            [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
            [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
        ])
        return R @ v

    def _get_obs(self):
        q = self.data.qpos[3:7]
        ang_vel = self._quat_rotate_inv(q, self.data.qvel[3:6])
        grav = self._quat_rotate_inv(q, np.array([0.0, 0.0, -1.0]))
        jpos = self.data.qpos[self.qpos_adr] - self.home
        jvel = self.data.qvel[self.dof_adr]
        cmd = np.zeros(13)  # stand = all-zero twist(3)/head(4)/body(6)
        return np.concatenate([ang_vel, grav, jpos, jvel, self._last_action, cmd]).astype(np.float32)

    # ---------------- reset ----------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        key = "STAND" if self.rng.random() < self.reset_stand_prob else "SIT"
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key)
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        self.ctrl.reset(self.data.qpos)
        self._last_action = np.zeros(self.n_act)
        self._steps = 0
        self._prev_trunk_z = self.data.xpos[self.trunk_id][2]
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    # ---------------- reward ----------------
    def _reward(self, action):
        d = self.data
        z = d.xpos[self.trunk_id][2]
        q = d.qpos[3:7]
        # posture_composite vs STAND target
        height_score = math.exp(-((z - _STAND_Z) / 0.03) ** 2)
        tilt_sq = 2.0 * (q[1] ** 2 + q[2] ** 2)
        upright_score = math.exp(-tilt_sq / (0.40 ** 2))
        jpos = d.qpos[self.qpos_adr]
        pose_err = ((jpos[self.leg_idx] - self.home[self.leg_idx]) ** 2).mean()
        pose_score = math.exp(-pose_err / (0.40 ** 2))
        head_err = ((jpos[self.neck_idx] - self.home[self.neck_idx]) ** 2).mean()
        head_score = math.exp(-head_err / (0.40 ** 2))
        composite = height_score * upright_score * pose_score * head_score

        # upright_linear (always-on): 1 - tilt (linear floor)
        upright_linear = max(0.0, 1.0 - math.sqrt(tilt_sq))

        # backward_support_margin: whole-body CoM vs heel line (soft wall)
        com = d.subtree_com[0]
        w, x, y, zz = q
        yaw = math.atan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz))
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        foot_xy = np.mean([d.site_xpos[s][:2] for s in self.foot_sites], axis=0)
        heel_xy = foot_xy - self.heel_offset * fwd
        margin = float(np.dot(com[:2] - heel_xy, fwd))
        back_pen = max(0.0, 0.005 - margin) if z > (_SIT_Z + 0.02) else 0.0

        # descent speed penalty (no crashing down)
        vz = (z - self._prev_trunk_z) / (self.dt * self.decimation)
        self._prev_trunk_z = z
        descent_pen = max(0.0, -vz - 0.5)  # penalize downward > 0.5 m/s

        action_rate = float(np.sum((action - self._last_action) ** 2))
        ang_vel_cost = float(np.sum(d.qvel[3:6] ** 2))

        r = (
            3.0 * composite
            + 2.5 * upright_linear
            - 2.0 * back_pen
            - 10.0 * descent_pen
            - 0.1 * action_rate
            - 0.05 * ang_vel_cost
            + self.alive_bonus
        )
        return r, {"composite": composite, "margin_mm": margin * 1000, "z": z}

    # ---------------- step ----------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target = self.home + action  # scale 1.0
        for i, name in enumerate(self.joint_names):
            self.ctrl.set_q_target(name, float(target[i]))
        for _ in range(self.decimation):
            self.ctrl.update()
            mujoco.mj_step(self.model, self.data)
        self._steps += 1
        self._last_action = action.copy()

        obs = self._get_obs()
        r, info = self._reward(action)
        z = self.data.xpos[self.trunk_id][2]
        q = self.data.qpos[3:7]
        tilt = math.sqrt(2.0 * (q[1] ** 2 + q[2] ** 2))
        fell = (z < _SIT_Z - 0.03) or (tilt > 1.2)
        terminated = bool(fell)
        if fell:
            r += self.fall_penalty
        truncated = self._steps >= self.max_steps
        info["fell"] = fell
        return obs, float(r), terminated, truncated, info

    def render(self):
        return None


if __name__ == "__main__":
    # quick smoke: random actions, report survival + obs shape
    env = DuckSitStandEnv(heel=False)
    obs, _ = env.reset(seed=0)
    print("obs shape", obs.shape, "n_act", env.n_act, "dt", env.dt, "max_steps", env.max_steps)
    tot = 0.0
    for i in range(200):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        tot += r
        if term or trunc:
            print(f"ep ended step {i} fell={info['fell']} z={info['z']:.3f} margin={info['margin_mm']:.1f}mm")
            obs, _ = env.reset()
    print("smoke OK, cumulative reward", round(tot, 2))