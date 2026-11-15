# The V2 feet in MuJoCo

Five variants share the XGO body, actuator order, ankle axes, named foot sites and named collision geoms:

| Variant | Contact geometry | Ankle inertia |
|---|---|---|
| `upstream` | Meshes in the stock upstream robot scene | Upstream measured/modelled values |
| `v2-stock` | Convex hull of each original V2 rigid foot and installed TPU pad | Same as upstream |
| `v2-heel5`, `v2-heel10`, `v2-heel15` | Convex hull of each Fusion 5, 10 or 15 mm foot and its installed, handed TPU pad | Base model plus estimated printed material at the heel |

Use `v2-stock` vs `v2-heel15` to compare two CAD designs with the same mesh generation procedure. Both render the actual printable foot and pad as separate visual meshes; each ankle uses one contact hull. MuJoCo's convex contact surface smooths over the screw recesses. TPU is modeled as a rigid shape. Contact parameters and surface compliance come from the environment. See the surface definitions in the environment for the contact approximation.

```python
from v2_feet import load_spec
spec = load_spec(robot_dir, 'v2-heel15')  # robot_dir contains scene.xml and assets/
model = spec.compile()
```

Use `mass_delta=False` to isolate geometry. To supply measured mass, weigh an old and new assembled foot with the same hardware and pass their difference in grams as `added_mass_g=...` (per foot). With the default estimate, the software divides upstream ankle mass by stock CAD volume, applies that effective density to the additional CAD volume, and uses the first and second moments of the extra material to update the ankle center of mass and inertia. Against the bundled carpet-standup robot the estimated additional masses per foot are **3.03 g (5 mm), 6.03 g (10 mm), and 9.03 g (15 mm)**. The estimate follows the base model’s ankle mass. Different infill, materials or hardware will change it. This method preserves the original measured/modelled inertia of the stock ankle.

**Registration.** Upstream and hardware have longitudinal foot length 54 mm, a 29.4 mm separation between the inner hub walls, and the same ankle axis offset. In the CAD STL frame the ankle axis is parallel to Y through X = −7 mm and Z = 16.5 mm. After mounting the pad (thicker on the inside), transform hardware coordinates to ankle-local millimeters with `(16.5-z, -s*(x+7), s*y-14.9665)`, where s=+1 on the left and −1 on the right. The input pad-to-foot alignment follows the screw centers in the Fusion assembly. Asset generation then converts millimeters to meters. All geometry is bundled locally for offline use.

**Verification.** From the lab root, run `python -m v2_feet.verify /path/to/src/mjlab_microduck/robot/xgoduck`. See `validation.json` for the actual checks against the bundled robot. In STAND, the registered stock feet and upstream sole heights agree within 0.012 mm. All three extensions preserve toe position and sole height within 0.02 mm. The rear envelope moves by the selected 5, 10, or 15 mm. In a second fixture, the trunk is rotated backward 12° and lowered to a 0.2 mm plane overlap; the simulator reports the extended feet contacting about 15 mm farther rearward than V2 stock. These are fixed-pose geometric and contact checks. Balance success needs a separate controlled policy evaluation. 

**Source and regeneration.** Hardware at [joshhiggins22/xgoduck_hardware, commit 52e237f](https://github.com/joshhiggins22/xgoduck_hardware/tree/52e237f0d53199b7ef697b7ce9e0662e7a662ffe) provides the stock STL files in `structure/V2_STANDARD_` and both new foot/pad STL pairs in `mods/v2-heel-fusion`. The [upstream XGO-Duck assets](https://github.com/LuwuDynamics/xgoduck_rl/tree/a4c78aacd5eaa724cf246fe12d24bfe1e837fa50/src/mjlab_microduck/robot/xgoduck) supply body frames and original inertia. See the hardware fork's `LICENSING.md` for design provenance. To rebuild the registered STLs and mass properties, use a separate environment with NumPy, trimesh and SciPy and run `python v2_feet/rebuild.py /path/to/checkout/of/xgoduck_hardware --output v2_feet`. The training environment only needs MuJoCo and NumPy to load these files.

The 5 and 10 mm test parts were exported by reopening the same Fusion master, setting `HeelExtension`, recomputing its feature history, and exporting each of the four solid bodies in local print coordinates. File hashes and Fusion body measurements are recorded in `short-heel-fusion.json`. The same registration and density calculation were used at all three lengths. The bundled base robot and its mass are from [the hardware owner’s training fork](https://github.com/joshhiggins22/xgoduck_rl/tree/c1091f58a87d635971a249950f429f27a1f25985). To regenerate the shorter variants, unpack the short-heel print archive and give the folder containing `heel5/` and `heel10/` to `rebuild.py --short-exports`.
