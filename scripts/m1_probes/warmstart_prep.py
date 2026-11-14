#!/usr/bin/env python3
"""Warm-start prep: (1) extract official ONNX actor -> rsl_rl actor_state_dict checkpoint,
(2) validate by replaying the EXTRACTED checkpoint in CPU MuJoCo (must still hold stand).
Uses the official home pose from ONNX metadata default_joint_pos."""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
import mujoco
import imageio.v2 as imageio
import onnx
from onnx import numpy_helper

HD = os.path.expanduser("~/duck-probe"); OUT = os.path.join(HD, "out"); os.makedirs(OUT, exist_ok=True)

# ---- Part 1: extract ONNX -> actor_state_dict ----
mo = onnx.load(os.path.join(HD, "official_sitstand.onnx"))
W = {w.name: numpy_helper.to_array(w) for w in mo.graph.initializer}
meta = {p.key: p.value for p in mo.metadata_props}
HOME = np.array([float(v) for v in meta["default_joint_pos"].split(",")], dtype=np.float32)
mean = W["obs_normalizer._mean"]; std = W["onnx::Div_24"]; var = std ** 2
def t(a): return torch.tensor(np.asarray(a), dtype=torch.float32)
actor_sd = {
    "obs_normalizer._mean": t(mean), "obs_normalizer._var": t(var),
    "obs_normalizer._std": t(std), "obs_normalizer.count": torch.tensor(1.0e6),
    "distribution.std_param": torch.full((14,), 0.5),
    "mlp.0.weight": t(W["mlp.0.weight"]), "mlp.0.bias": t(W["mlp.0.bias"]),
    "mlp.2.weight": t(W["mlp.2.weight"]), "mlp.2.bias": t(W["mlp.2.bias"]),
    "mlp.4.weight": t(W["mlp.4.weight"]), "mlp.4.bias": t(W["mlp.4.bias"]),
    "mlp.6.weight": t(W["mlp.6.weight"]), "mlp.6.bias": t(W["mlp.6.bias"]),
}
ck = {"actor_state_dict": actor_sd, "iter": 0,
      "infos": {"warm_start_from": "official_sitstand.onnx", "home_rad": HOME.tolist()}}
torch.save(ck, os.path.join(HD, "official_warmstart.pt"))
print("saved official_warmstart.pt | actor keys:", list(actor_sd.keys()))
print("official HOME(rad):", np.round(HOME, 4))

# ---- Part 2: replay the EXTRACTED checkpoint ----
sd = torch.load(os.path.join(HD, "official_warmstart.pt"), map_location="cpu", weights_only=False)["actor_state_dict"]
W0,b0=sd["mlp.0.weight"],sd["mlp.0.bias"]; W2,b2=sd["mlp.2.weight"],sd["mlp.2.bias"]
W4,b4=sd["mlp.4.weight"],sd["mlp.4.bias"]; W6,b6=sd["mlp.6.weight"],sd["mlp.6.bias"]
omean=sd["obs_normalizer._mean"][0].numpy(); ostd=sd["obs_normalizer._std"][0].numpy()
cmask=ostd<1e-6
def actor(obs):
    n=(obs-omean)/np.where(cmask,1.0,ostd); n=np.where(cmask,0.0,n)
    x=torch.from_numpy(n).float()
    x=F.elu(x@W0.T+b0); x=F.elu(x@W2.T+b2); x=F.elu(x@W4.T+b4)
    return (x@W6.T+b6).numpy()
def qri(q,v):
    w,x,y,z=q; tt=np.array([x,y,z])
    return v*(2*w*w-1)-np.cross(tt,v)*(2*w)+tt*np.dot(tt,v)*2

m=mujoco.MjModel.from_xml_path(os.path.join(HD,"model","scene.xml")); d=mujoco.MjData(m)
m.vis.global_.offwidth=960; m.vis.global_.offheight=720
fid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"floor"); m.geom_solref[fid]=[0.02,1.0]
ren=mujoco.Renderer(m,480,640)
cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat=np.array([0.,0.,0.10]); cam.distance=0.5; cam.azimuth=90; cam.elevation=-12
DT=m.opt.timestep; DECIM=5; NCTRL=500; KT=[0.0,1.0,2.5,4.0,4.99]
def obs_of(last):
    q=d.qpos[3:7].copy()
    return np.concatenate([qri(q,d.qvel[3:6].copy()),qri(q,np.array([0.,0.,-1.])),
                           d.qpos[7:21]-HOME,d.qvel[6:20].copy(),last,
                           np.zeros(3),np.zeros(4),np.zeros(6)]).astype(np.float32)
def run(start,tag):
    mujoco.mj_resetDataKeyframe(m,d,mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,start)); mujoco.mj_forward(m,d)
    last=np.zeros(14,dtype=np.float32); zs=[];pit=[];rol=[];frames=[];kf={}
    for i in range(NCTRL):
        a=np.clip(actor(obs_of(last)),-1,1); d.ctrl[:]=HOME+a
        for _ in range(DECIM): mujoco.mj_step(m,d)
        last=a; t=i*DT*DECIM; q=d.qpos[3:7]; g=qri(q,np.array([0.,0.,-1.]))
        zs.append(float(d.qpos[2])); pit.append(float(np.degrees(np.arctan2(g[0],-g[2])))); rol.append(float(np.degrees(np.arctan2(g[1],-g[2]))))
        ren.update_scene(d,camera=cam); frames.append(ren.render())
        for kt in KT:
            if abs(t-kt)<DT*DECIM/2 and kt not in kf: kf[kt]=ren.render().copy()
    zs=np.array(zs)
    imageio.mimsave(os.path.join(OUT,f"warmstart_{tag}.mp4"),frames,fps=30,macro_block_size=1)
    for kt,img in kf.items(): imageio.imwrite(os.path.join(OUT,f"warmstart_{tag}_t{kt:.1f}.png"),img)
    print(f"=== warmstart {tag} === z_start={zs[0]:.4f} z_end={zs[-1]:.4f} z_max={zs.max():.4f} "
          f"pitch_end={pit[-1]:+.1f} roll_end={rol[-1]:+.1f} reached_stand={bool(zs.max()>0.135)}")
    return {"z_end":float(zs[-1]),"z_max":float(zs.max()),"pitch_end":float(pit[-1]),"roll_end":float(rol[-1])}
res={tag:run(s,tag) for s,tag in [("STAND","hold_stand"),("SIT","sit_to_stand")]}
json.dump(res,open(os.path.join(OUT,"warmstart_summary.json"),"w"),indent=2)
print("VALIDATION: extracted checkpoint replayed; hold_stand should stay ~0.137 m")
