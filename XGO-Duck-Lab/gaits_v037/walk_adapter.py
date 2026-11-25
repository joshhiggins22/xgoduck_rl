"""Command-space adapter for the XGO-Duck 0.3.7 Walk observation."""
from collections.abc import Mapping

import numpy as np


BOUNDS = {"vx_gain": (1.0, 1.6), "vy_per_vx": (-0.25, 0.25), "wz_per_vx": (-1.0, 1.0)}
IDENTITY = {"vx_gain": 1.0, "vy_per_vx": 0.0, "wz_per_vx": 0.0}
FORWARD_LIMIT = np.float32(0.20)
YAW_LIMIT = np.float32(0.35)
STRAFE_LIMIT = np.float32(0.08)
SPEED_KNOTS = np.array([0.10, 0.15, 0.20], dtype=np.float32)


def _values(params):
    if isinstance(params, Mapping):
        if params.keys() == {"speed_profile"}:
            rows = params["speed_profile"]
            if len(rows) != len(SPEED_KNOTS):
                raise ValueError("The speed profile needs three rows for 0.10, 0.15 and 0.20 m/s")
            rows = np.asarray([_values(row) for row in rows], dtype=np.float32)
            if rows.shape != (3, 3):
                raise ValueError("Each profile row needs three coefficients")
            return rows
        if params.keys() != BOUNDS.keys():
            raise ValueError(f"Expected parameters {tuple(BOUNDS)}")
        params = [params[name] for name in BOUNDS]
    return np.asarray(params, dtype=np.float32)


def coefficients(params):
    """Validate and canonicalize either a single correction or all three knot rows."""
    values = _values(params)
    if values.shape not in ((3,), (3, 3)) or not np.isfinite(values).all():
        raise ValueError("Expected one or three rows of three finite adapter coefficients")
    for i, (name, (low, high)) in enumerate(BOUNDS.items()):
        if (values[..., i] < np.float32(low)).any() or (values[..., i] > np.float32(high)).any():
            raise ValueError(f"{name} must be in [{low}, {high}]")
    return values


def adapt_twist(twist, params):
    """Return the policy command after any upstream heading-hold correction."""
    vx, vy, wz = np.asarray(twist, dtype=np.float32)
    values = _values(params)
    if values.ndim == 2:
        low, middle, high = SPEED_KNOTS
        if vx <= low:
            values = values[0]
        elif vx >= high:
            values = values[2]
        else:
            i = int(vx >= middle)
            blend = (vx - SPEED_KNOTS[i]) / (SPEED_KNOTS[i + 1] - SPEED_KNOTS[i])
            values = (np.float32(1) - blend) * values[i] + blend * values[i + 1]
    gain, lateral, yaw = values
    forward = min(max(vx, np.float32(0)), FORWARD_LIMIT)
    turn_weight = max(np.float32(0), np.float32(1) - abs(wz) / YAW_LIMIT)
    strafe_weight = max(np.float32(0), np.float32(1) - abs(vy) / STRAFE_LIMIT)
    weight = forward * turn_weight * strafe_weight
    return np.array([vx + (np.float32(gain) - np.float32(1)) * weight,
                     vy + np.float32(lateral) * weight,
                     wz + np.float32(yaw) * weight], dtype=np.float32)
