#!/usr/bin/env python3
"""Replay the OFFICIAL deployed xgoduck_sitstand.onnx in CPU MuJoCo (M1).
Same 61-D obs + normalizer + ELU MLP as the training actor; weights pulled straight
from the ONNX initializers. Uses the OFFICIAL home pose (ONNX metadata default_joint_pos,
else runtime HOME_DEG). Validates the warm-start source: does the official policy stand,
and does it show the backward-fall at the end of the head-lift?"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
import mujoco
import imageio.v2 as imageio
import onnx
from onnx import numpy_helper

HD = os.path.expanduser("~/duck-probe")
OUT = os.path.join(HD, "out"); os.makedirs(OUT, exist_ok=True)
m_onnx = onnx.load(os.path.join(HD, "official_sitstand.onnx"))
W = {w.name: numpy_helper.to_array(w) for w in m_onnx.graph.initializer}
meta = {p.key: p.value for p in m_onnx.metadata_props}
print("ONNX metadata keys:", list(meta.keys()))

# official home pose (radians, policy order)
if "default_joint_pos" in meta:
    HOME = np.array([float(v) for v in meta["default_joint_pos"].split(",")], dtype=np.float32)
    print("home from ONNX metadata")
else:
    HOME = np.deg2rad(np.array([0,-5,-22,0,22, 20,20,0,0, 0,5,22,0,-22], dtype=np.float32))
    print("home from runtime HOME_DEG (±22 deg)")
print("HOME(rad):", np.round(HOME, 4))

omean = W["obs_normalizer._mean"][0]
ostd = W["onnx::Div_24"][0]
const_mask = ostd < 1e-6
print("std zeros (const dims):", np.where(const_mask)[0].tolist())
W0,b0 = torch.tensor(W["mlp.0.weight"]), torch.tensor(W["mlp.0.bias"])
W2,b2 = torch.tensor(W["mlp.2.weight"]), torch.tensor(W["mlp.2.bias"])
W4,b4 = torch.tensor(W["mlp.4.weight"]), torch.tensor(W["mlp.4.bias"])
W6,b6 = torch.tensor(W["mlp.6.weight"]), torch.tensor(W["mlp.6.bias"])

def actor(obs):
    norm = (obs - omean) / np.where(const_mask, 1.0, ostd)
    norm = np.where(const_mask, 0.0, norm)
    x = torch.from_numpy(norm).float()
    x = F.elu(x @ W0.T + b0); x = F.elu(x @ W2.T + b2)
    x = F.elu(x @ W4.T + b4)
    return (x @ W6.T + b6).numpy()

def qri(q, v):
    w,x,y,z = q; t = np.array([x,y,z])
    return v*(2*w*w-1) - np.cross(t,v)*(2*w) + t*np.dot(t,v)*2

m = mujoco.MjModel.from_xml_path(os.path.join(HD, "model", "scene.xml"))
d = mujoco.MjData(m)
m.vis.global_.offwidth=960; m.vis.global_.offheight=720
fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
m.geom_solref[fid]=[0.02,1.0]
ren = mujoco.Renderer(m,480,640)
cam = mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat=np.array([0.,0.,0.10]); cam.distance=0.5; cam.azimuth=90; cam.elevation=-12
DT=m.opt.timestep; DECIM=5; NCTRL=600   # 6 s
KT=[0.0,1.0,2.0,3.0,4.0,5.0,5.99]

def obs_of(last):
    q=d.qpos[3:7].copy()
    return np.concatenate([qri(q,d.qvel[3:6].copy()), qri(q,np.array([0.,0.,-1.])),
                           d.qpos[7:21]-HOME, d.qvel[6:20].copy(), last,
                           np.zeros(3),np.zeros(4),np.zeros(6)]).astype(np.float32)

def run(start, tag):
    mujoco.mj_resetDataKeyframe(m,d,mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,start))
    mujoco.mj_forward(m,d)
    last=np.zeros(14,dtype=np.float32); zs=[]; pitch=[]; roll=[]; frames=[]; kf={}
    for i in range(NCTRL):
        a=np.clip(actor(obs_of(last)),-1,1)
        d.ctrl[:]=HOME+a
        for _ in range(DECIM): mujoco.mj_step(m,d)
        last=a; t=i*DT*DECIM
        q=d.qpos[3:7]; g=qri(q,np.array([0.,0.,-1.]))
        zs.append(float(d.qpos[2])); pitch.append(float(np.degrees(np.arctan2(g[0],-g[2]))))
        roll.append(float(np.degrees(np.arctan2(g[1],-g[2]))))
        ren.update_scene(d,camera=cam); frames.append(ren.render())
        for kt in KT:
            if abs(t-kt)<DT*DECIM/2 and kt not in kf: kf[kt]=ren.render().copy()
    zs=np.array(zs); pitch=np.array(pitch); roll=np.array(roll)
    print(f"  [{tag}] t,z,pitch,roll samples:")
    for i in range(0,NCTRL,100):
        print(f"    t={i*DT*DECIM:.1f}s z={zs[i]:.3f} pitch={pitch[i]:+.1f} roll={roll[i]:+.1f}")
    mp4=os.path.join(OUT,f"official_{tag}.mp4")
    imageio.mimsave(mp4,frames,fps=30,macro_block_size=1)
    for kt,img in kf.items(): imageio.imwrite(os.path.join(OUT,f"official_{tag}_t{kt:.1f}.png"),img)
    s={"tag":tag,"start":start,"z_start":float(zs[0]),"z_end":float(zs[-1]),
       "z_max":float(zs.max()),"reached_stand_0.135":bool(zs.max()>0.135),
       "pitch_end_deg":float(pitch[-1]),"pitch_min_deg":float(pitch.min()),
       "pitch_max_deg":float(pitch.max()),
       "roll_end_deg":float(roll[-1]),"roll_min_deg":float(roll.min()),"roll_max_deg":float(roll.max()),
       "video":mp4}
    print(f"=== official {tag} ===")
    for k,v in s.items():
        if k!="video": print(f"   {k}: {v}")
    return s,zs.tolist(),pitch.tolist()

res={}
for start,tag in [("SIT","sit_to_stand"),("STAND","hold_stand")]:
    s,z,p=run(start,tag); res[tag]={"summary":s,"z":z,"pitch_deg":p}
json.dump(res,open(os.path.join(OUT,"official_summary.json"),"w"),indent=2)
print("Wrote official_summary.json + videos")
