"""Build registered meshes and additive ankle mass properties from the V2 source STLs."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

PARTS = Path("structure/V2_STANDARD_")
EXTENDED = Path("mods/v2-heel-fusion")
# Rigid V2 mounting face is print Z=0. The ankle axis passes through X=-7,
# Z=16.5 mm. The two inner hub walls are 29.4 mm apart. At the upright
# reference pose +CAD X is forward, +Y is robot-left, and +Z is up.
HUB_WALL_MM = 14.9665
# Measured from the four-hole centres in the Fusion assembly.
PAD_SHIFT_MM = 0.1258
STOCK_FILES = {
    "left": ("V2_STANDARD_LEFT_FOOT.stl", "V2_STANDARD_2x_FOOT_BOTTOM_TPU_02.stl"),
    "right": ("V2_STANDARD_RIGHT_FOOT.stl", "V2_STANDARD_2x_FOOT_BOTTOM_TPU_01.stl"),
}


def get_parts(hw, side, variant, fusion_dir=None):
    sign = 1 if side == "left" else -1
    if variant == "v2-stock":
        foot_file, pad_file = (hw / PARTS / f for f in STOCK_FILES[side])
    else:
        directory = fusion_dir if fusion_dir is not None else hw / EXTENDED
        foot_file = directory / f"{side.title()}_foot.stl"
        pad_file = directory / f"{side.title()}_TPU.stl"
    foot, pad = [trimesh.load_mesh(p) for p in (foot_file, pad_file)]
    if variant == "v2-stock" and side == "left":
        # Both factory stock pad STL instances are identical. The left pad is
        # mounted with a half-turn relative to the right. The long, newly handed
        # pads encode handedness explicitly and must use their matching file.
        pad.apply_transform(np.diag([-1.0, -1.0, 1.0, 1.0]))
    pad.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))  # mating planes face each other
    pad.apply_translation([0, -sign * PAD_SHIFT_MM, 0])
    cad_to_ankle = [
        [0, 0, -1, 16.5],
        [-sign, 0, 0, -sign * 7],
        [0, sign, 0, -HUB_WALL_MM],
        [0, 0, 0, 1],
    ]
    for m in (foot, pad):
        m.apply_transform(cad_to_ankle)
        m.apply_scale(0.001)
        if not m.is_volume:
            raise ValueError(f"{side} {variant}: requires closed solid source")
    return foot, pad


def sum_properties(parts):
    mesh = trimesh.util.concatenate(parts)
    # density=1 kg/m^3 => scale raw values by measured effective density later.
    mp = mesh.mass_properties
    c = mp.center_mass
    inertia_origin = mp.inertia + mp.mass * (np.eye(3) * np.dot(c, c) - np.outer(c, c))
    return np.r_[mp.volume, mp.volume * c, inertia_origin.reshape(-1)]


def generate(hw, out):
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    data = {}
    for side in ("left", "right"):
        props = {}
        for variant in ("v2-stock", "v2-heel15"):
            parts = get_parts(hw, side, variant)
            props[variant] = sum_properties(parts)
            for tag, m in zip(("foot", "pad"), parts):
                m.export(assets / f"{variant}_{side}_{tag}.stl")
            hull = trimesh.util.concatenate(parts).convex_hull
            hull.export(assets / f"{variant}_{side}_hull.stl")
        diff = props["v2-heel15"] - props["v2-stock"]
        data[side] = dict(
            stock_volume_m3=float(props["v2-stock"][0]),
            added_volume_m3=float(diff[0]),
            added_com_m=(diff[1:4] / diff[0]).tolist(),
            added_inertia_at_origin_per_kg_m2=(diff[4:].reshape(3, 3) / diff[0]).tolist(),
        )
    (out / "properties.json").write_text(json.dumps(data, indent=2) + "\n")


def generate_short(hw, out, short_exports):
    """Use unmodified Fusion local-coordinate STL exports for each parameter value."""
    for length in (5, 10):
        variant = f"v2-heel{length}"
        props = {}
        for side in ("left", "right"):
            stock = sum_properties(get_parts(hw, side, "v2-stock"))
            parts = get_parts(hw, side, variant, short_exports / f"heel{length}")
            diff = sum_properties(parts) - stock
            props[side] = dict(
                stock_volume_m3=float(stock[0]),
                added_volume_m3=float(diff[0]),
                added_com_m=(diff[1:4] / diff[0]).tolist(),
                added_inertia_at_origin_per_kg_m2=(diff[4:].reshape(3, 3) / diff[0]).tolist(),
            )
            hull = trimesh.util.concatenate(parts).convex_hull
            for role, mesh in zip(("foot", "pad", "hull"), (*parts, hull)):
                mesh.export(out / "assets" / f"{variant}_{side}_{role}.stl")
        (out / f"properties-heel{length}.json").write_text(json.dumps(props, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hardware", type=Path)
    ap.add_argument("--output", default=Path(__file__).resolve().parent, type=Path)
    ap.add_argument(
        "--short-exports", type=Path, help="Folder containing heel5/ and heel10/ Fusion part STLs."
    )
    a = ap.parse_args()
    generate(a.hardware, a.output)
    if a.short_exports:
        generate_short(a.hardware, a.output, a.short_exports)
