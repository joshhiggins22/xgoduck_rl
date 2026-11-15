"""Use the XGO Arduino runtime's observation and ONNX processing in CPU MuJoCo.

Coordinates of the simulated encoders are already logical joint radians. The
MCU's motor direction signs apply only at the physical encoder/bus interface.
"""

from importlib import import_module
from pathlib import Path
import sys

import numpy as np


JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


def _runtime_modules(directory):
    directory = Path(directory).resolve()
    names = ("rl_core", "proc", "policy")
    for name in names:
        path = directory / (name + ".py")
        if not path.is_file():
            raise FileNotFoundError(f"Missing XGO runtime file: {path}")
        prior = sys.modules.get(name)
        if prior is not None and Path(prior.__file__).resolve() != path:
            raise RuntimeError(f"{name} already loaded from another runtime: {prior.__file__}")
    sys.path.insert(0, str(directory))
    try:
        return tuple(import_module(name) for name in names)
    finally:
        sys.path.remove(str(directory))


class SimFeedback:
    """100 Hz simulated encoders and IMU passed through the actual receive filter."""

    def __init__(self, model, runtime_dir):
        self.model = model
        self.core, proc, self.runtime_policy = _runtime_modules(runtime_dir)
        self._filter = proc.SignalFilter()
        self.joint_names = JOINT_NAMES
        self._joints = np.array([model.joint(name).id for name in JOINT_NAMES])
        self._qadr = model.jnt_qposadr[self._joints]
        self._vadr = model.jnt_dofadr[self._joints]
        self._imu_site = model.site("imu").id
        self._last_t = None
        self.state = None

    def _process(self, data, acceleration):
        core = self.core
        angles = np.zeros(len(core.MOTOR_IDS))
        rates = np.zeros_like(angles)
        angles[core.POLICY_INDEX] = np.rad2deg(data.qpos[self._qadr])
        rates[core.POLICY_INDEX] = np.rad2deg(data.qvel[self._vadr])
        values = self._filter.update(
            acceleration,
            np.rad2deg(data.sensor("imu_ang_vel").data),
            angles,
            rates,
            timestamp=float(data.time),
            imu_timestamp=float(data.time),
        )
        self.state = core.State(
            acceleration, [], angles, values["vels_deg"], values["grav"], values["gyro_rad"]
        )
        self._last_t = float(data.time)

    def reset(self, data):
        self._filter.reset()
        # Initialize like a supported robot before torque-on. The free-fall
        # accelerometer in a just-reset, slightly elevated MJCF reads zero.
        rotation = data.site_xmat[self._imu_site].reshape(3, 3)
        self._process(data, rotation.T @ self.model.opt.gravity)

    def sample(self, data):
        if self._last_t is None:
            return self.reset(data)
        if data.time < self._last_t:
            raise ValueError("Simulation time moved backward; call reset after an environment reset.")
        if data.time > self._last_t:
            # MuJoCo returns specific force (+g at rest). Firmware exports a
            # body-axis gravity reading (-g at rest). Gyro already uses IMU axes.
            self._process(data, -data.sensor("imu_accel").data.copy())

    sample_feedback = sample

    def obs(self, prev_action, cmd=None, head=None, home_deg=None):
        if self.state is None:
            raise RuntimeError("Call reset(data) before reading simulated feedback.")
        return self.core.build_obs(
            self.state,
            np.asarray(prev_action, dtype=np.float32),
            np.zeros(3) if cmd is None else cmd,
            np.zeros(4) if head is None else head,
            home_deg=self.core.HOME_DEG if home_deg is None else home_deg,
        )


class PolicyActions:
    """Runtime home and action smoothing, for training with sim-to-runtime parity."""

    def __init__(self, feedback, scale=1.0):
        self.core = feedback.core
        self.scale = float(scale)
        self.alphas = np.full(14, self.core.ACTION_ALPHA, dtype=np.float32)
        self.alphas[5:9] = self.core.ACTION_ALPHA_HEAD
        self.reset()

    def reset(self):
        self.raw_action = np.zeros(14, dtype=np.float32)
        self.filtered_action = np.zeros_like(self.raw_action)

    def targets(self, raw_action):
        raw = np.asarray(raw_action, dtype=np.float32)
        if raw.shape != (14,) or not np.isfinite(raw).all():
            raise ValueError("Expected 14 finite action values in policy joint order.")
        self.raw_action[:] = raw
        self.filtered_action *= self.alphas
        self.filtered_action += raw * (1.0 - self.alphas)
        motor_deg = self.core.action_to_deg(self.filtered_action, action_scale=self.scale)
        return np.deg2rad(np.asarray(motor_deg)[self.core.POLICY_INDEX])


class StockPolicy:
    """Official in-graph normalization, 50 Hz ONNX and runtime action filters."""

    def __init__(self, model, runtime_dir, onnx_path=None, scale=1.0):
        self.feedback = SimFeedback(model, runtime_dir)
        self.joint_names = JOINT_NAMES
        self.policy = self.feedback.runtime_policy.Policy()
        path = Path(onnx_path) if onnx_path else Path(runtime_dir) / "xgoduck_sitstand.onnx"
        if not path.is_file():
            raise FileNotFoundError(path)
        ok, message = self.policy.load(path)
        if not ok:
            raise ValueError(message)
        metadata = self.policy.session.get_modelmeta().custom_metadata_map
        names = metadata.get("joint_names")
        if names and tuple(names.split(",")) != JOINT_NAMES:
            raise ValueError("ONNX joint order differs from the XGO runtime.")
        self.policy.action_scale = float(scale)

    @property
    def raw_action(self):
        return self.policy.prev_action.copy()

    def reset(self, data):
        self.feedback.reset(data)
        self.policy.reset_action_filter()

    def sample(self, data):
        self.feedback.sample(data)

    sample_feedback = sample

    def targets(self, data=None, command=None, head=None):
        if data is not None:
            self.sample(data)
        st = self.feedback.state
        if st is None:
            raise RuntimeError("Call reset(data) before inference.")
        filtered = self.policy.infer(st.grav, st.gyro_rad, st.angles, st.vels, command, head)
        motor_deg = self.policy.action_to_deg(filtered)
        return np.deg2rad(np.asarray(motor_deg)[self.feedback.core.POLICY_INDEX])
