"""Native CPU XGO sit-to-stand environment with the 61-value Arduino actor interface."""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco as mj
import numpy as np

from bam.model import load_model
from bam.mujoco import MujocoController
from floor_surfaces import CARPET_IMPEDANCE, HARDWOOD_IMPEDANCE, CarpetField
from stock_policy import JOINT_NAMES, PolicyActions, SimFeedback

ROOT = Path(__file__).resolve().parent
from v2_feet import VARIANTS as FEET

SURFACES = ("hard", "carpet", "mixed", "hardwood", "residential-carpet")
DT = 0.005
DECIMATION = 4


class DuckSitStandEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        foot="upstream",
        surface="hard",
        start="mixed",
        episode_s=6.0,
        seed=None,
        perturb=True,
        mass_delta=True,
        runtime_dir=None,
    ):
        super().__init__()
        if foot not in FEET or surface not in SURFACES or start not in ("sit", "stand", "mixed"):
            raise ValueError("Unknown foot, surface or starting pose")
        if episode_s <= 0:
            raise ValueError("episode_s must be positive")
        self.foot, self.surface, self.start = foot, surface, start
        self.crouch_depth = None
        self.max_steps = round(episode_s / (DT * DECIMATION))
        self.perturb, self.initial_seed = perturb, seed
        robot = ROOT / "robot"
        if foot == "upstream":
            spec = mj.MjSpec.from_file(str(robot / "scene.xml"))
        else:
            from v2_feet import load_spec

            spec = load_spec(robot, foot, mass_delta=mass_delta)
        spec.option.timestep = DT
        self.joint_names = list(JOINT_NAMES)
        servo = load_model(str(robot / "params/1910_m6.json"))
        servo.actuator.kp = 5.0
        for actuator in spec.actuators:
            actuator.set_to_motor()
            actuator.forcelimited = True
            lim = 8.0 * servo.kt.value / servo.R.value
            actuator.forcerange = [-lim, lim]
            actuator.gear = [1.0, 0, 0, 0, 0, 0]
        for joint in spec.joints:
            if joint.name in JOINT_NAMES:
                joint.armature = float(servo.actuator.get_extra_inertia())
                joint.damping = np.zeros_like(joint.damping)
                joint.frictionloss = 0.0
        if surface == "residential-carpet":
            for side in ("left", "right"):
                spec.add_pair(name=f"{side}_carpet", geomname1="floor", geomname2=f"{side}_foot_collision",
                              condim=3, solref=[.060, 1.], solimp=CARPET_IMPEDANCE,
                              friction=[.79, .79, .005, .0001, .0001])
        self.model = spec.compile()
        self.data = mj.MjData(self.model)
        self.ctrl = MujocoController(
            servo, self.joint_names, self.model, self.data, vin_drop_gain=0.1, vin_min=7.0
        )
        self.feedback = SimFeedback(self.model, runtime_dir or ROOT / "runtime")
        self.actions = PolicyActions(self.feedback)
        self.home = np.deg2rad(self.feedback.core.HOME_DEG).astype(float)
        self.qpos_adr = self.ctrl.qpos_indexes
        self.dof_adr = self.ctrl.dof_indexes
        self.qrange = self.model.jnt_range[self.ctrl.joint_indexes]
        self.floor = self.model.geom("floor").id
        self.trunk = self.model.body("trunk_base").id
        self.soles = [self.model.geom(s + "_foot_collision").id for s in ("left", "right")]
        self.carpet_pairs = ([self.model.pair(f"{s}_carpet").id for s in ("left", "right")]
                             if surface == "residential-carpet" else None)
        self.carpet_field = None
        self.dt = DT
        self.decimation = DECIMATION
        self.observation_space = spaces.Box(-np.inf, np.inf, (61,), np.float32)
        # Raw learned outputs are radian offsets; hardware applies physical joint limits downstream.
        self.action_space = spaces.Box(-8.0, 8.0, (14,), np.float32)
        self.model.geom_priority[self.floor] = 2

    def set_crouch_depth(self, depth):
        """Training-only reset curriculum. Evaluation always uses full seated starts."""
        self.crouch_depth = float(np.clip(depth, 0.0, 1.0))

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self.initial_seed
        self.initial_seed = None
        super().reset(seed=seed)
        rng = self.np_random
        pose = (options or {}).get("start", self.start)
        if pose == "mixed":
            pose = "sit" if rng.random() < 0.65 else "stand"
        if pose not in ("sit", "stand"):
            raise ValueError("Unknown reset pose")
        self.pose = pose
        self.material = self.surface if self.surface != "mixed" else rng.choice(["hard", "carpet"])
        soft = self.material == "carpet"
        m, d = self.model, self.data
        if self.material in ("hardwood", "residential-carpet"):
            # Isolate material sampling from initial pose, so a matched seed
            # starts with the same physical perturbation across these surfaces.
            surface_rng = np.random.default_rng(rng.integers(2**63))
            if self.material == "hardwood":
                tau, mu = surface_rng.uniform(.010, .014), surface_rng.uniform(.55, .80)
                impedance = HARDWOOD_IMPEDANCE
                self.contact_params = dict(tau_s=float(tau), friction=float(mu), solimp=list(impedance))
            else:
                self.carpet_field = CarpetField(surface_rng)
                tau, mu = self.carpet_field.tau, self.carpet_field.mu
                impedance = CARPET_IMPEDANCE
                self.contact_params = self.carpet_field.describe()
        else:
            tau = rng.uniform(0.035, 0.065) if soft else 0.015
            mu = rng.uniform(0.55, 1.1) if soft else 0.75
            impedance = [0.70, 0.93, 0.004, 0.5, 2.0] if soft else [0.9, 0.95, 0.001, 0.5, 2.0]
            self.contact_params = {"tau_s": float(tau), "friction": float(mu)}
        m.geom_solref[self.floor] = [tau, 1.0]
        m.geom_solimp[self.floor] = impedance
        m.geom_friction[self.floor] = [mu, 0.005, 0.0001]
        mj.mj_resetDataKeyframe(m, d, m.key("STAND").id)
        # Training sit pose; the companion viewer's SIT key has a different head and knee pose.
        if pose == "sit":
            d.qpos[self.qpos_adr] = self.home
            d.qpos[self.qpos_adr[[1, 2, 3, 4, 10, 11, 12, 13]]] = [
                0,
                -0.4079,
                1.35,
                0,
                0,
                0.4079,
                -1.35,
                0,
            ]
            depth = 1.0
            if self.crouch_depth is not None and rng.random() < 0.8:
                depth = rng.uniform(max(0.15, self.crouch_depth - 0.18), self.crouch_depth)
                q = d.qpos[self.qpos_adr]
                d.qpos[self.qpos_adr] = self.home + depth * (q - self.home)
            d.qpos[2] = 0.139158 + depth * (0.085868 - 0.139158)
        if self.perturb:
            pitch, roll = np.deg2rad(rng.uniform(-2.0, 2.0, 2))
            d.qpos[3:7] = [
                np.cos(pitch / 2) * np.cos(roll / 2),
                np.cos(pitch / 2) * np.sin(roll / 2),
                np.sin(pitch / 2) * np.cos(roll / 2),
                -np.sin(pitch / 2) * np.sin(roll / 2),
            ]
            d.qvel[3:5] = rng.normal(0.0, 0.025, 2)
        if self.carpet_field is not None:
            self._update_carpet()
        mj.mj_forward(m, d)
        dist = min(mj.mj_geomDistance(m, d, self.floor, g, 0.2, None) for g in self.soles)
        d.qpos[2] += 0.0005 - dist
        mj.mj_forward(m, d)
        self.ctrl.reset(d.qpos)
        self.ctrl.last_ts = d.time - DT
        self.ctrl.model.actuator.q_target_smooth = d.qpos[self.qpos_adr].copy()
        self.model.dof_damping[self.dof_adr] = 0.0
        self.model.dof_frictionloss[self.dof_adr] = 0.0
        self.ctrl._prev_motor_torque.fill(0)
        self._delayed = d.qpos[self.qpos_adr].copy()
        mj.mj_forward(m, d)
        self.feedback.reset(d)
        self.actions.reset()
        self._last_action = np.zeros(14, np.float32)
        self._prev_z = float(d.xpos[self.trunk, 2])
        self.steps = self.upright_streak = 0
        return self.feedback.obs(self._last_action), {
            "start": pose,
            "surface": self.material,
            **self.contact_params,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        return self.advance(self.actions.targets(action), action)

    def _update_carpet(self):
        # Refresh positions for the current qpos before building this step's contacts.
        # Explicit pairs leave the underlying plane and robot self contacts alone.
        mj.mj_kinematics(self.model, self.data)
        self.carpet_field.update_pairs(self.model, self.carpet_pairs, self.data.geom_xpos[self.soles, :2])

    def advance(self, target, action, sample_callback=None):
        d, m = self.data, self.model
        # Encoder travel spans roughly one revolution. MuJoCo enforces the mechanical stops.
        target = np.clip(target, -np.pi, np.pi)
        for i in range(DECIMATION):
            self.ctrl.q_target[:] = self._delayed
            self._delayed = target
            self.ctrl.update()
            if self.carpet_field is not None:
                self._update_carpet()
            mj.mj_step(m, d)
            if i % 2 == 1:
                mj.mj_forward(m, d)
                self.feedback.sample(d)
                if sample_callback is not None:
                    sample_callback(d)
        self.steps += 1
        z = float(d.xpos[self.trunk, 2])
        tilt = float(np.arccos(np.clip(d.xmat[self.trunk, 8], -1.0, 1.0)))
        knee = float(np.max(np.abs(d.qpos[self.qpos_adr[[3, 12]]])))
        stable = z >= 0.124 and tilt < np.deg2rad(15.0) and knee < 0.40
        self.upright_streak = self.upright_streak + 1 if stable else 0
        fell = tilt > np.deg2rad(65.0) or z < 0.052
        # Rear support is derived from present foot-floor contacts in the torso heading.
        heading = d.xmat[self.trunk].reshape(3, 3)[:2, 0].copy()
        heading /= max(np.linalg.norm(heading), 1e-8)
        rear_pts = [
            np.dot(c.pos[:2], heading)
            for c in d.contact
            if self.floor in c.geom and any(g in c.geom for g in self.soles) and c.dist <= 0.001
        ]
        margin = (
            float(np.dot(d.subtree_com[self.trunk, :2], heading) - min(rear_pts)) if rear_pts else None
        )
        height = np.exp(-(((z - 0.139158) / 0.03) ** 2))
        upright = np.exp(-((tilt / 0.40) ** 2))
        pose_error = np.mean((d.qpos[self.qpos_adr] - self.home) ** 2)
        upright_reward = 3.0 * height * upright * np.exp(-pose_error / 0.18) + 2.5 * max(
            0.0, 1.0 - tilt
        )
        rate = float(np.mean((action - self._last_action) ** 2))
        vz = (z - self._prev_z) / (DT * DECIMATION)
        reward = upright_reward + float(stable) - 0.12 * rate - max(0.0, -vz - 0.45) * 3.0
        if margin is not None and z > 0.105:
            reward -= min(1.0, 40 * max(0.0, 0.005 - margin))
        if fell:
            reward -= 20.0
        self._prev_z = z
        self._last_action = np.asarray(action, dtype=np.float32).copy()
        obs = self.feedback.obs(self._last_action)
        if not np.isfinite(obs).all():
            raise FloatingPointError("Non-finite sensor state")
        info = dict(
            fell=bool(fell),
            z=z,
            tilt_deg=float(np.rad2deg(tilt)),
            knee_rad=knee,
            margin_mm=None if margin is None else 1000 * margin,
            upright=bool(stable),
            upright_streak_s=self.upright_streak * DT * DECIMATION,
        )
        return obs, float(reward), bool(fell), self.steps >= self.max_steps, info
