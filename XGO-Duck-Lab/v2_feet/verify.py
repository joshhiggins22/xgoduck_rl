"""Verify geometry, registrations, ankle inertias and floor contacts in compiled MuJoCo."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from . import VARIANTS, HEEL_LENGTHS, load_spec


def compiled_vertices(model, data, side):
    gid = model.geom(f"{side}_foot_collision").id
    mesh = model.geom_dataid[gid]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    rotation = data.geom_xmat[gid].reshape(3, 3)
    return model.mesh_vert[start : start + count] @ rotation.T + data.geom_xpos[gid]


def measure(robot_dir, variant, include_mass=True):
    model = load_spec(robot_dir, variant, mass_delta=include_mass).compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    actions = [model.actuator(i).name for i in range(model.nu)]
    initial_height = data.qpos[2]
    results = {}
    for pitch in (0, -12):
        pitch_radians = np.radians(pitch)
        data.qpos[3:7] = [np.cos(pitch_radians / 2), 0, np.sin(pitch_radians / 2), 0]
        data.qpos[2] = initial_height
        mujoco.mj_forward(model, data)
        points = {s: compiled_vertices(model, data, s) for s in ("left", "right")}
        result = {
            s: {
                "toe_x_mm": p[:, 0].max() * 1000,
                "heel_x_mm": p[:, 0].min() * 1000,
                "lowest_z_mm": p[:, 2].min() * 1000,
            }
            for s, p in points.items()
        }
        # Pose fixture: lower the root until the lowest vertex overlaps a level
        # plane by 0.2 mm, then inspect the simulator's actual contact manifold.
        data.qpos[2] -= min(p[:, 2].min() for p in points.values()) + 0.0002
        mujoco.mj_forward(model, data)
        for s in points:
            floor_id, foot_id = model.geom("floor").id, model.geom(f"{s}_foot_collision").id
            contact_x = [c.pos[0] for c in data.contact if {c.geom1, c.geom2} == {floor_id, foot_id}]
            if not contact_x:
                raise AssertionError(f"{variant}, pitch={pitch}: no {s} floor contact")
            result[s]["rear_floor_contact_x_mm"] = min(contact_x) * 1000
        results[pitch] = result
    return dict(
        nq=model.nq,
        nu=model.nu,
        actions=actions,
        fixture=results,
        ankle_masses_g={
            s: model.body_mass[model.body("ankle_" + s).id] * 1000 for s in ("left", "right")
        },
    )


def verify(robot_dir):
    cases = {v: measure(robot_dir, v) for v in VARIANTS}
    upstream, stock = cases["upstream"], cases["v2-stock"]
    pure_geometry = measure(robot_dir, "v2-heel15", False)
    for variant, length in HEEL_LENGTHS.items():
        heel = cases[variant]
        assert heel["actions"] == stock["actions"] == upstream["actions"]
        assert (heel["nu"], heel["nq"]) == (upstream["nu"], upstream["nq"])
        for side in ("left", "right"):
            a, b = stock["fixture"][0][side], heel["fixture"][0][side]
            assert abs(a["lowest_z_mm"] - upstream["fixture"][0][side]["lowest_z_mm"]) < 0.1
            assert abs(a["lowest_z_mm"] - b["lowest_z_mm"]) < 0.02
            assert abs(a["toe_x_mm"] - b["toe_x_mm"]) < 0.02
            assert abs((a["heel_x_mm"] - b["heel_x_mm"]) - length) < 0.05
            displacement = (
                stock["fixture"][-12][side]["rear_floor_contact_x_mm"]
                - heel["fixture"][-12][side]["rear_floor_contact_x_mm"]
            )
            assert length - 1 < displacement < length + 1, (variant, side, displacement)
            assert heel["ankle_masses_g"][side] > stock["ankle_masses_g"][side]
            assert pure_geometry["ankle_masses_g"][side] == stock["ankle_masses_g"][side]
    return {
        "mujoco": mujoco.__version__,
        "variants": cases,
        "heel_geometry_only_masses_g": pure_geometry["ankle_masses_g"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_dir", type=Path)
    print(json.dumps(verify(parser.parse_args().robot_dir), indent=2))
