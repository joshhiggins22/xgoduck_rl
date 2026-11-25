# Hardwood and residential-carpet simulation

Use `hardwood` and `residential-carpet` with the stock, 5 mm or 10 mm V2 heels. Both floors are level at zero height. A matched seed gives the same initial robot perturbation on each floor and a repeatable material field for each heel. The earlier `hard`, `carpet` and `mixed` presets keep their original physics and seeds.

## How much a foot sinks

The 5 mm robot model weighs about 7.89 N. A 4 N test load approximates half its weight, and 8 N approximates its full weight on one foot. The table shows minimum, median and maximum static indentation across the tested sole geometries and floor locations.

| Load on one foot | Hardwood, mm | Residential carpet, mm |
| ---: | ---: | ---: |
| 2 N | 0.007 / 0.009 / 0.014 | 2.02 / 2.86 / 3.55 |
| 4 N | 0.013 / 0.015 / 0.024 | 2.57 / 3.52 / 4.31 |
| 8 N | 0.025 / 0.029 / 0.043 | 3.08 / 4.15 / 5.04 |

From `XGO-Duck-Lab/gaits_v037`, run `python heel_profiles/calibrate_surfaces.py` in the lab environment to reproduce the test. A 0.40 kg fixture holds each compiled left and right collision hull in its neutral orientation and permits only vertical motion. Known downward loads act for two seconds each. Each cell contains 108 samples across three heel sizes, both feet, three material seeds and six XY locations. All samples reached static balance within 2% of the applied normal force and showed increasing indentation with increasing load.

## Contact assumptions

- **Hardwood.** Each episode samples sliding friction from 0.55–0.80 and contact time constant from 10–14 ms. Impedance increases from 0.93 to 0.985 over 0.8 mm. The plane has no seams or height variation.
- **Residential carpet.** Each sole samples a stationary material field at its world position every 5 ms. Two smooth waves with randomized directions, phases and wavelengths of 0.16–0.27 m and 0.32–0.58 m describe changes in local support. Across possible seeds and positions, sliding friction spans 0.67–0.91, contact time constant spans 44–76 ms, and impedance rises from 0.45 to 0.95 over 5.5–8.5 mm. Both feet sample the same field. The nonlinear, damped contact lets each sole sink under load and makes further compression progressively harder. [MuJoCo contact solver](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)

The model combines pile and cushion into a single effective support layer. Individual fibers, directional nap, snagging, lasting footprints, seams and worn-through spots are outside its scope. The Carpet and Rug Institute gives a typical residential cushion thickness range of roughly 6–11 mm. The force–depth and friction ranges above are simulation assumptions. Measurements on the actual flooring and assembled soles would establish how closely this preset represents the user's carpet. [Carpet and Rug Institute](https://carpet-rug.org/carpet-for-homes/selecting-the-right-carpet/)
