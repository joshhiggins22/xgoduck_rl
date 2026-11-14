#!/usr/bin/env python3
"""Replay the trained sitstand policy (model_6999.pt) in CPU MuJoCo on the M1.
Faithful-ish: exact 61-D obs layout + obs normalizer + ELU MLP from the checkpoint,
action = HOME_FRAME + 1.0*action, decimation 5 (100 Hz). Caveat: raw scene.xml uses
XML position servos, not the BAM motor model used in training (close approximation).
Commands all-zero = STAND. Renders side-view mp4 + key frames + trunk-z timeline."""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
import mujoco
import imageio.v2 as imageio

HOME_DIR = os.path.expanduser("~/duck-probe")
MODEL = os.path.join(HOME_DIR, "model", "scene.xml")
CKPT = os.path.join(HOME_DIR, "model_6999.pt")
OUT = os.path.join(HOME_DIR, "out")
os.makedirs(OUT, exist_ok=True)

# HOME_FRAME.joint_pos in obs/ctrl joint order (0-4 L leg, 5-8 head, 9-13 R leg)
HOME = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530,
                 0.3491, 0.3491, 0.0, 0.0,
                 0.0, 0.0873, 0.4579, 0.0049, -0.4530], dtype=np.float32)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["actor_state_dict"]
W0, b0 = sd["mlp.0.weight"], sd["mlp.0.bias"]
W2, b2 = sd["mlp.2.weight"], sd["mlp.2.bias"]
W4, b4 = sd["mlp.4.weight"], sd["mlp.4.bias"]
W6, b6 = sd["mlp.6.weight"], sd["mlp.6.bias"]
omean = sd["obs_normalizer._mean"][0].numpy()
ovar = sd["obs_normalizer._var"][0].numpy()
ostd = np.sqrt(ovar + 1e-8)
const_mask = ovar < 1e-8   # dims constant in training (e.g. zero-padded body cmd)
print("obs mean range:", omean.min(), omean.max(), "| std range:", ostd.min(), ostd.max(),
      "| const dims:", np.where(const_mask)[0].tolist())

def actor(obs):
    norm = (obs - omean) / ostd
    norm = np.where(const_mask, 0.0, norm)
    norm = np.clip(norm, -5.0, 5.0)
    x = torch.from_numpy(norm).float()
    x = F.elu(x @ W0.T + b0)
    x = F.elu(x @ W2.T + b2)
    x = F.elu(x @ W4.T + b4)
    return (x @ W6.T + b6).numpy()

def quat_rot_inv(q, v):
    w, x, y, z = q
    # rotate vector v by inverse of quaternion q (w,x,y,z)
    t = np.array([x, y, z])
    a = v * (2 * w * w - 1)
    b = np.cross(t, v) * (2 * w)
    c = t * np.dot(t, v) * 2
    return a - b + c

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)
m.vis.global_.offwidth = 960; m.vis.global_.offheight = 720
floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
m.geom_solref[floor_id] = [0.02, 1.0]   # nominal ground
renderer = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = np.array([0.0, 0.0, 0.10]); cam.distance = 0.50
cam.azimuth = 90.0; cam.elevation = -12.0
DT = m.opt.timestep; DECIM = 5
NCTRL = 500          # 5 s at 100 Hz
KEY_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 4.99]

def compute_obs(last_action):
    quat = d.qpos[3:7].copy()
    angvel = quat_rot_inv(quat, d.qvel[3:6].copy())
    grav = quat_rot_inv(quat, np.array([0.0, 0.0, -1.0]))
    jpos_rel = d.qpos[7:21] - HOME
    jvel = d.qvel[6:20].copy()
    return np.concatenate([angvel, grav, jpos_rel, jvel, last_action,
                           np.zeros(3), np.zeros(4), np.zeros(6)]).astype(np.float32)

def run(start_key, tag):
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, start_key))
    mujoco.mj_forward(m, d)
    g0 = quat_rot_inv(d.qpos[3:7].copy(), np.array([0, 0, -1.0]))
    print(f"[{tag}] reset gravity_b={np.round(g0,3)} trunk_z={d.qpos[2]:.4f}")
    last_action = np.zeros(14, dtype=np.float32)
    zs, acts, frames, keyframes = [], [], [], {}
    for i in range(NCTRL):
        obs = compute_obs(last_action)
        a = actor(obs)
        a = np.clip(a, -1.0, 1.0)
        d.ctrl[:] = HOME + a
        for _ in range(DECIM):
            mujoco.mj_step(m, d)
        last_action = a
        t = i * DT * DECIM
        zs.append(float(d.qpos[2])); acts.append(float(np.abs(a).mean()))
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render())
        for kt in KEY_TIMES:
            if abs(t - kt) < DT * DECIM / 2 and kt not in keyframes:
                keyframes[kt] = renderer.render().copy()
    zs = np.array(zs)
    mp4 = os.path.join(OUT, f"policy_{tag}.mp4")
    imageio.mimsave(mp4, frames, fps=30, macro_block_size=1)
    for kt, img in keyframes.items():
        imageio.imwrite(os.path.join(OUT, f"policy_{tag}_t{kt:.1f}.png"), img)
    summ = {"tag": tag, "start": start_key,
            "z_start": float(zs[0]), "z_end": float(zs[-1]),
            "z_max": float(zs.max()), "z_min": float(zs.min()),
            "reached_stand_0.135": bool(zs.max() > 0.135),
            "mean_abs_action": float(np.mean(acts)),
            "video": mp4}
    print(f"=== policy {tag} ===")
    for k, v in summ.items():
        if k != "video": print(f"   {k}: {v}")
    return summ, zs.tolist()

results, tl = {}, {}
for start, tag in [("SIT", "sit_to_stand"), ("STAND", "hold_stand")]:
    s, z = run(start, tag); results[tag] = s; tl[tag] = z
json.dump({"results": results, "trunk_z": tl}, open(os.path.join(OUT, "policy_summary.json"), "w"), indent=2)
print("Wrote policy_summary.json + videos")
