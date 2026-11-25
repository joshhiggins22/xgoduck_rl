# v0.3.7 forward Run policy for V2 feet with 5 mm heels

`xgoduck_run.onnx` is an optional fast-walk policy for the XGO-Duck v0.3.7 controller. It builds on the [5 mm heel Walk policy](../v037-v2-heel5-forward-walk/) with a fitted forward-command adjustment and bounded changes to the leg swing. The v0.3.7 loader accepted its 61 observations and 14 joint actions and completed 12 warm-up inferences. Run has been tested in simulation only.

| SHA-256 | Model |
| --- | --- |
| `882084ecaea8290a83672a08757eac6d8ed2048e60d148a489110e2e8fa9e28d` | Run |
| `15e8b8ea6fb69a98fdf3c028c3899859bfd2416bca6b8e2cea0b2ce409d5233c` | Walk used for comparison |

## Simulation results

Both policies received the same 0.20 m/s forward command from a standing start. The 3 m times average 12 paired trials per surface. The 50 m speeds average six paired trials per surface. The finish line measures forward progress along the initial heading.

| Surface | Walk time for 3 m | Run time for 3 m | Walk speed over 50 m | Run speed over 50 m | Run mean peak sideways deviation over 50 m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hard floor | 12.76 s | 9.88 s | 0.234 m/s | 0.306 m/s | 1.56 m |
| Carpet | 11.67 s | 9.36 s | 0.254 m/s | 0.317 m/s | 0.77 m |

The duck stopped upright in 76 course and abrupt-release trials, 48 trials with simulated servo variation, and 24 trials that switched between Walk and Run. Across those trials, maximum displacement after release was 6.7 cm and the longest settling time was 0.77 s. Release timing starts when the controller receives a zero forward command.

The simulated mode switch preserved the previous raw action and filtered action, with heading hold active. Releasing the stick sent zero forward command while balance control continued.
