"""Load XGO-Duck MuJoCo scenes with registered V2 foot geometry."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HEEL_LENGTHS = {"v2-heel5": 5, "v2-heel10": 10, "v2-heel15": 15}
VARIANTS = ("upstream", "v2-stock", *HEEL_LENGTHS)
HERE = Path(__file__).resolve().parent


def _numbers(values):
    return " ".join(f"{v:.13g}" for v in values)


def _parallel_axis(position):
    return np.eye(3) * np.dot(position, position) - np.outer(position, position)


def _add_heel_inertia(elem, prop, measured_g):
    mass = float(elem.attrib["mass"])
    delta = (
        mass * prop["added_volume_m3"] / prop["stock_volume_m3"]
        if measured_g is None
        else float(measured_g) / 1000
    )
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("Added mass must be a finite, nonnegative value in grams per foot")
    x, y, z, xy, xz, yz = map(float, elem.attrib["fullinertia"].split())
    old_inertia = np.array([[x, xy, xz], [xy, y, yz], [xz, yz, z]])
    old_com = np.fromstring(elem.attrib["pos"], sep=" ")
    new_mass = mass + delta
    new_com = (mass * old_com + delta * np.asarray(prop["added_com_m"])) / new_mass
    new_inertia = (
        old_inertia
        + mass * _parallel_axis(old_com)
        + delta * np.asarray(prop["added_inertia_at_origin_per_kg_m2"])
        - new_mass * _parallel_axis(new_com)
    )
    elem.set("mass", f"{new_mass:.13g}")
    elem.set("pos", _numbers(new_com))
    elem.set("fullinertia", _numbers(new_inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))


def load_spec(
    robot_dir, variant="upstream", scene_name="scene.xml", *, mass_delta=True, added_mass_g=None
):
    """Return a fresh MjSpec. robot_dir contains scene.xml, robot XML and assets/.

    Extended feet default to a CAD-volume estimate of added mass. Supply the
    measured mass difference (grams per assembled foot) or mass_delta=False
    for a geometry-only comparison.
    """
    base = Path(robot_dir).resolve()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown foot variant {variant!r}; choose one of {VARIANTS}")
    if added_mass_g is not None and (variant not in HEEL_LENGTHS or not mass_delta):
        raise ValueError("added_mass_g applies only to an extended heel with mass_delta enabled")
    if variant == "upstream":
        return mujoco.MjSpec.from_file(str(base / scene_name))
    scene = ET.parse(base / scene_name).getroot()
    include = scene.find("include")
    if include is None or len(scene.findall("include")) != 1:
        raise ValueError("Expected a stock XGO scene with one robot include")
    robot = ET.parse(base / include.attrib["file"]).getroot()
    assets = robot.find("asset")
    compiler = robot.find("compiler")
    # Resolve relative STL paths once so the resulting spec compiles from any CWD.
    mesh_dir = base / compiler.attrib.get("meshdir", "")
    compiler.attrib.pop("meshdir", None)
    for mesh in assets.findall("mesh"):
        if "file" in mesh.attrib:
            mesh.set("file", str(mesh_dir / mesh.attrib["file"]))
    props_name = (
        f"properties-heel{HEEL_LENGTHS[variant]}.json"
        if variant in ("v2-heel5", "v2-heel10")
        else "properties.json"
    )
    props = json.loads((HERE / props_name).read_text())
    for side in ("left", "right"):
        body = robot.find(f".//body[@name='ankle_{side}']")
        if body is None:
            raise ValueError("Expected XGO ankle bodies, check the robot asset version")
        geom = body.find(f"geom[@name='{side}_foot_collision']")
        # The floor collision uses one convex hull, as in the upstream model.
        # Preserve the original geom's name, class, and contact configuration.
        old = body.find(f"geom[@class='visual'][@mesh='{side}_ankle']")
        body.remove(old)
        for role in ("foot", "pad", "hull"):
            name = f"{variant}_{side}_{role}"
            ET.SubElement(assets, "mesh", name=name, file=str(HERE / "assets" / f"{name}.stl"))
            if role == "hull":
                geom.set("mesh", name)
            else:
                color = "0.88 0.59 0.16 1" if role == "foot" else "0.16 0.19 0.21 1"
                ET.SubElement(
                    body,
                    "geom",
                    {
                        "name": f"{side}_{role}_visual",
                        "type": "mesh",
                        "mesh": name,
                        "class": "visual",
                        "rgba": color,
                    },
                )
        if variant in HEEL_LENGTHS and mass_delta:
            _add_heel_inertia(body.find("inertial"), props[side], added_mass_g)
    index = list(scene).index(include)
    scene.remove(include)
    for i, node in enumerate(robot):
        scene.insert(index + i, node)
    return mujoco.MjSpec.from_string(ET.tostring(scene, encoding="unicode"))
