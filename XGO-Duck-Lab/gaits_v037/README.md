# Walk and Run on hardwood and residential carpet

These tests replay the v0.3.7 controller's ONNX inference, command trim, heading hold and action filters in the native MuJoCo lab. Each heel size uses the corresponding V2 foot collision geometry and mass. Both feet interact with the same repeatable carpet material field. The simulator samples contact every 5 ms, filters feedback every 10 ms and runs the actor every 20 ms.

Use the lab's Python environment from this directory. The frozen outputs live in `../runs/v037-hardwood-residential-carpet`. The following commands measure the floor response, replay all heel sizes on the two floors and verify the exported policies.

```sh
profiles=../runs/v037-hardwood-residential-carpet
python heel_profiles/calibrate_surfaces.py
python audit_heel_profiles.py --source "$profiles" --out /tmp/heel-course.json --seed 5104428 --trials 12 --touch 8 --long 6
python heel_profiles/perturb_check.py --directory "$profiles" --out /tmp/heel-hardware.json --seed 9148212 --trials 64 --run-policy-trim
python heel_profiles/verify_exports.py "$profiles" ../runs/v037-v2-heel5-forward-walk/walk.onnx /tmp/heel-exports.json
python run_switches.py --walk "$profiles/5mm/walk.onnx" --run "$profiles/5mm/run.onnx" --foot v2-heel5 --surfaces hardwood residential-carpet --out /tmp/heel-switches.json
```

To start a new population fit from the previously released 5 mm policies, run the following command. It fits Walk and Run for all three geometries, stores candidate graphs, and ranks the candidates' upright movement and stops. Independent course and hardware-variation evaluations measure the frozen files.

```sh
python train_heel_profiles.py --out /tmp/heel-fit --seed 521125 --population 100 --refine 60
```

The fitted corrections apply to forward motion and adjust the leg targets. The actor observes joint state, gravity, gyro and requested motion. The training uses the reported flooring only through the simulated dynamics. The live operator selects the heels and can use either gait on either surface.

[Floor parameters and load response](surfaces.md) · [Released policies and results](../runs/v037-hardwood-residential-carpet/)
