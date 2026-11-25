# XGO-Duck v0.3.7 heel policies

Each heel size has a Walk and a Run policy trained across both floors. Select the heels physically fitted to the duck. The initial selection is 5 mm. Run uses the faster policy for those same heels. All performance figures below come from physics simulation.

| Heels | Floor | Walk, 3 m | Run, 3 m | Walk, 50 m | Run, 50 m | Speed gain over 50 m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| factory | hardwood | 12.97 s | 8.69 s | 0.230 m/s | 0.346 m/s | +50.0% |
| factory | residential-carpet | 12.06 s | 7.47 s | 0.249 m/s | 0.365 m/s | +46.6% |
| 5mm | hardwood | 12.18 s | 8.74 s | 0.247 m/s | 0.344 m/s | +39.2% |
| 5mm | residential-carpet | 11.76 s | 8.04 s | 0.257 m/s | 0.372 m/s | +45.1% |
| 10mm | hardwood | 12.88 s | 9.03 s | 0.233 m/s | 0.319 m/s | +36.9% |
| 10mm | residential-carpet | 11.83 s | 7.53 s | 0.255 m/s | 0.387 m/s | +51.5% |

The benchmark commands full forward from a standing start and measures progress along the original heading. Walk and Run use the same start and floor realization. At each finish the stick returns to neutral. Other trials use partial stick or sudden releases at different points in the step.

The nominal suite reached the 50 m finish 96 times. Across 576 trials there were 0 falls and 576 sustained stops. The largest displacement from release was 9.0 cm. The longest settling time on a successful stop was 0.84 s. The largest sideways deviation was 9.31 m.

With simulated motor strength, encoder, IMU and leg-delay differences, the policies stayed upright in 1152 of 1152 trials and completed 1152 sustained stops. A stop requires at least two continuous seconds upright with planar motion below 2 cm/s and yaw below 8 degrees/s in every 0.2-second window.

While changing Walk and Run at the control boundary, 108 of 108 trials stayed upright and 108 achieved a sustained stop.

The residential-carpet preset models a level, compressible floor. It sinks 2.6–4.3 mm under a 4 N single-foot load across the simulated locations and heels. The stiffness and friction of the physical carpet have not been measured.

The accompanying results file includes individual heel/floor speeds, drift, tilt, stops by IMU calibration state and SHA-256 digests.

## Controller and simulation

The controller patch applies to the v0.3.7 app with the earlier Run checkbox installed. The six ONNX files belong under `python/models/heels/<heel>/` in that app. The independent Run checkbox uses the selected heel pair. The selector remembers the last heel selection after restarting. The full checksum manifest and simulation results accompany the model files.

For straight forward Run without saved IMU calibration, the controller supplies the Run actor with its own gravity estimate. The same estimator is available in the stress test. The controller keeps using its primary sensor estimate for heading and fall recovery.

[Controller and UI patch](heel-selector-v037.patch) · [Simulation and reproduction commands](../../gaits_v037/) · [Detailed validation](validation.json)
