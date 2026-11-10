# Horizontal Wind Force Correction

User reports vertical disturbance despite horizontal-only settings. A fresh CPU reproduction has no configured wind at startup, accepts only `[X,Y,0]`, yet predicts vertical acceleration `0.037781... m/s²` from the unchanged PINN. The wrapper applies all three predicted components; this reproduces the symptom without stale settings. The user's horizontal-disturbance requirement now applies to actual wind force, not only wind velocity inputs.

- [x] Write failing regressions for actual MuJoCo Fz, previous vertical EMA state, unchanged horizontal filtering and real CPU predictions (stationary and vertical-motion states).
- [x] Project predicted residuals to world XY before filtering, clear only the filter's Z state, preserve raw 3D diagnostics and unchanged PINN weights/architecture.
- [x] Update old force expectations and documentation; run full offline tests and verify the real CPU reproduction after correction.

This is an explicit application-layer projection P=diag(1,1,0), not a retraining or assertion that horizontal airflow physically cannot induce vertical forces. Do not change UAV gravity, rotor thrust, vertical controller or altitude movement controls. No PX4 edits or real SITL launch. Backup: /tmp/wind-horizontal-force-before.

Verification: all four new cases reproduced the failure before correction. Final full suite 136 passed in 16.76s, no skips. Regression checks actual xfrc_applied Fz and exported force/filtered diagnostics, clearing old Z EMA without changing the independently expected XY update; raw three-axis predictions remain intact. Real CPU checkpoint cases pass with both zero and nonzero vertical vehicle velocity. No SITL flight verification performed.
