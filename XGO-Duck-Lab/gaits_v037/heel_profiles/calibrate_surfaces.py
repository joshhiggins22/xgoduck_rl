"""Measure simulated single-sole load and depth over sampled floor locations."""
from collections import defaultdict
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from duck_env import DT, DuckSitStandEnv
from floor_surfaces import CARPET_IMPEDANCE


def make_press(env, sole):
    """Hold the compiled hull in its neutral orientation on a vertical slide."""
    m, d = env.model, env.data
    mesh = m.geom_dataid[sole]
    start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
    vertices = m.mesh_vert[start:start+count].astype(float) @ d.geom_xmat[sole].reshape(3,3).T
    vertices[:, :2] -= vertices[:, :2].mean(axis=0)
    vertices[:, 2] -= vertices[:, 2].min()
    vertex_text = " ".join(f"{x:.12g}" for x in vertices.ravel())
    xml = f"""<mujoco><option timestep="{DT}" gravity="0 0 0"/>
    <asset><mesh name="sole" vertex="{vertex_text}"/></asset>
    <worldbody><geom name="floor" type="plane" size="0 0 .02" priority="2"/>
    <body name="press" pos="0 0 .001"><joint type="slide" axis="0 0 1"/>
    <inertial mass=".4" pos="0 0 0" diaginertia=".001 .001 .001"/>
    <geom type="mesh" mesh="sole"/></body></worldbody></mujoco>"""
    return mj.MjModel.from_xml_string(xml)


def loaded_depth(press, solref, solimp, friction, load):
    press.geom_solref[0] = solref
    press.geom_solimp[0] = solimp
    press.geom_friction[0,0] = friction
    data = mj.MjData(press)
    data.xfrc_applied[1,2] = -load
    mj.mj_step(press, data, nstep=400)
    mj.mj_forward(press,data)
    f = np.empty(6)
    normal = 0.
    for idx in range(data.ncon):
        mj.mj_contactForce(press,data,idx,f)
        normal += f[0]
    assert abs(normal-load) < .02*load, (load,normal)
    assert abs(data.qvel[0]) < 2e-5
    return -1000*(.001 + data.qpos[0])


def main():
    depths = defaultdict(list)
    for foot in ("v2-stock","v2-heel5","v2-heel10"):
        for surface in ("hardwood","residential-carpet"):
            env = DuckSitStandEnv(foot=foot,surface=surface,start="stand",perturb=False)
            env.reset(seed=411)
            fixtures = [make_press(env,sole) for sole in env.soles]
            for seed in (411, 2879, 19807):
                env.reset(seed=seed)
                for xy in ((0.,0.),(.065,.04),(.17,-.09),(.36,.07),(.57,-.12),(.88,.04)):
                    floor = env.floor
                    if env.carpet_field is None:
                        ref = env.model.geom_solref[floor]
                        imp = env.model.geom_solimp[floor]
                        mu = env.model.geom_friction[floor,0]
                    else:
                        tau, width, mu = env.carpet_field.at(xy)
                        ref, imp = [tau,1.], list(CARPET_IMPEDANCE)
                        imp[2] = width
                    for press in fixtures:
                        values = [loaded_depth(press,ref,imp,mu,load) for load in (2.,4.,8.)]
                        assert 0 < values[0] < values[1] < values[2]
                        for load, value in zip((2,4,8), values):
                            depths[(surface,foot,load)].append(value)
    print("Surface / sole / load (N) / depth min, median, max (mm)")
    for key, values in depths.items():
        print(*key, *(round(v,3) for v in np.percentile(values,[0,50,100])))
    print("Ranges across all soles")
    for surface in ("hardwood","residential-carpet"):
        for load in (2,4,8):
            values = [v for (s,_,n), entries in depths.items() if (surface,load)==(s,n) for v in entries]
            print(surface,load,len(values),*(round(v,3) for v in np.percentile(values,[0,50,100])))


if __name__ == "__main__":
    main()
