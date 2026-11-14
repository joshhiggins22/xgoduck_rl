#!/usr/bin/env python3
"""Settle-probe: does the carpet-compliance DR (soft ground solref) make the
XGO-Duck's standing pose sink/bounce? Holds the STAND keyframe on nominal ground
(solref 0.02) vs the DR soft extreme (0.06), logs trunk z + ground reaction, and
renders a side-view mp4 + key frames for each. CPU MuJoCo only (M1)."""
import os, json
import numpy as np
import mujoco
import imageio.v2 as imageio

HOME = os.path.expanduser("~/duck-probe")
MODEL = os.path.join(HOME, "model", "scene.xml")
OUT = os.path.join(HOME, "out")
os.makedirs(OUT, exist_ok=True)

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)
m.vis.global_.offwidth = 960
m.vis.global_.offheight = 720

floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
stand_key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
print("floor solref (xml default):", m.geom_solref[floor_id])
print("model timestep:", m.opt.timestep, "| nq:", m.nq, "| nu:", m.nu)

renderer = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = np.array([0.0, 0.0, 0.10])
cam.distance = 0.50
cam.azimuth = 90.0
cam.elevation = -12.0

DT = m.opt.timestep
NSTEP = 600            # 2.4 s of sim
RENDER_EVERY = 2       # 125 fps sim -> 300 frames
FPS = 30               # ~4x slow-mo playback
KEY_TIMES = [0.0, 0.4, 0.8, 1.2, 1.8, 2.39]

def ground_reaction():
    """Total normal force from floor contacts (N)."""
    f = np.zeros(6)
    tot = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 == floor_id or c.geom2 == floor_id:
            mujoco.mj_contactForce(m, d, i, f)
            tot += f[0]
    return tot

def run(tc, tag):
    mujoco.mj_resetDataKeyframe(m, d, stand_key)
    m.geom_solref[floor_id] = [tc, 1.0]
    mujoco.mj_forward(m, d)
    z0 = float(d.qpos[2])
    zs, grs, frames = [], [], []
    keyframes = {}
    for i in range(NSTEP):
        mujoco.mj_step(m, d)             # ctrl held at STAND keyframe values
        t = i * DT
        zs.append(float(d.qpos[2]))
        grs.append(ground_reaction())
        if i % RENDER_EVERY == 0:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())
        for kt in KEY_TIMES:
            if abs(t - kt) < DT / 2 and kt not in keyframes:
                renderer.update_scene(d, camera=cam)
                keyframes[kt] = renderer.render().copy()
    zs = np.array(zs); grs = np.array(grs)
    mp4 = os.path.join(OUT, f"stand_{tag}.mp4")
    imageio.mimsave(mp4, frames, fps=FPS, macro_block_size=1)
    for kt, img in keyframes.items():
        imageio.imwrite(os.path.join(OUT, f"stand_{tag}_t{kt:.1f}.png"), img)
    summary = {
        "tag": tag, "solref_timeconst": tc,
        "z_start": z0, "z_end": float(zs[-1]),
        "z_min": float(zs.min()), "z_max": float(zs.max()),
        "z_mean_last_half": float(zs[len(zs)//2:].mean()),
        "z_std_last_half": float(zs[len(zs)//2:].std()),
        "drop_from_start_mm": float((z0 - zs[-1]) * 1000),
        "gr_mean_N": float(grs.mean()), "gr_min_N": float(grs.min()),
        "gr_max_N": float(grs.max()),
        "weight_N": float(m.body_subtreemass[1] * 9.81),
        "ncon_end": int(d.ncon),
        "video": mp4,
    }
    print(f"\n=== {tag} (solref tc={tc}) ===")
    for k, v in summary.items():
        if k != "video":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return summary, zs.tolist()

results = {}
timelines = {}
for tc, tag in [(0.02, "nominal"), (0.06, "carpet_extreme")]:
    s, z = run(tc, tag)
    results[tag] = s
    timelines[tag] = z

with open(os.path.join(OUT, "settle_summary.json"), "w") as f:
    json.dump({"results": results, "trunk_z": timelines}, f, indent=2)
print("\nWrote", os.path.join(OUT, "settle_summary.json"))
print("Videos:", [results[t]["video"] for t in results])
