# X4-950 PINN pilot v1 model card

## Intended use and interface

Default `pinn-x4` wind-increment backend in this project's MuJoCo/PX4 SITL,
selected after user approval following the24-run comparison. Legacy `pinn`
and `analytic` remain explicit fallbacks.
No checkpoint is installed over the old model. No PX4 code or control-parameter
changes are required. Invalid candidate metadata is rejected before connecting.
Fallback means explicitly selecting `--wind-backend pinn`, not silent swapping.

Bundled checkpoint: `src/wind_field_simulation/assets/checkpoints/x4_950_wind_v1/candidate.pt`.
Original trial: `outputs/pinn/x4_950_pilot_v1/training_corrected/candidate.pt`.
The four bundled checkpoint/reference files are byte-identical copies of the
verified trial/reference, not a retrained or renamed model kind.
SHA256: `d7fb92c96b275245047e76cfdb4317d4ca72c5ef81529c854f686e221ad5996b`.
Kind: `x4_950_wind_increment_xy_v1`.

Core inputs: `[v_air_B(3), omega_B(3)]`, m/s and rad/s. Core output: complete
three-axis reference acceleration B, m/s². B is FLU at root O; W is the existing
ENU-like world. Unit WXYZ quaternion maps B to W. It is not a CoM model.

Runtime evaluates the same-state wind/no-wind pair in one CPU float32 batch:
`R [g(Rᵀ(v_O-w), omega) - g(Rᵀ v_O, omega)]`. The adapter returns all three raw
world components for diagnostics. The existing app projects XY, runs EMA=.05,
limits each axis to ±6m/s², multiplies by total mass, applies root force only.
No full no-wind drag or additional torque is applied.

Zero wind produces exactly zero raw increment. Applying zero after nonzero
wind retains the existing XY EMA decay; switching wind OFF resets the filter
and force. Raw Z can be nonzero; actual applied Fz stays exactly zero.

## Frozen domain and assumptions

- Horizontal resultant wind 0–3m/s, wz=0; [3,3,0] is rejected, never clipped.
- Fixed initial X4-950 + Airbot Play geometry, no payload, total mass 8.0004kg.
  Real simulated arm holds its initial position with existing joint control;
  current joint deviation is logged. Geometry is not updated from live joints.
- Offline scan root XY speed<=3m/s, vertical speed±1m/s, roll/pitch±20°,
  all yaw, angular-speed norm<=1.5rad/s. This is an inspection domain, not a
  demonstrated flight envelope. Beyond-domain states are logged; the wind
  domain is strictly enforced. No claim of accurate state extrapolation.
- Component quadratic drag teacher: 29 deduplicated proxies, rho=1.225kg/m³,
  Cd axes=1. These are simulation assumptions, not measured parameters.
- Excludes induced rotor flow, wake interference, active arm geometry, extra
  payload, ground/wall effects. MuJoCo fluid forces remain disabled.
- Frozen reference hash:
  `613e8c2840f524ec5caf742a7d03a00dff930fc006669612953d1abc897f5e1a`.

## Training and known limitations

3000 nominal synthetic same-state pairs, seed7; grouped 2000/500/500 diagnostic
split. No flight data or old PINN outputs are teacher labels. Fixed nonnegative
unit-axis quadratic prior + 6→64→64→3 Tanh residual; train-only normalization.
Full and XY increment supervised losses plus a soft positive-work penalty on
complete acceleration only at omega=0. Seed42, CPU deterministic,200epochs.

Heldout XY RMSE .000655/.000617m/s²; nonzero-wind normalized vector error1.282%.
Zero-wind raw increment exact0 for3000states. These are pilot diagnostics,
not formal final-test or real aerodynamic-accuracy results. See
[pilot report](../X4_PINN_PILOT_2026-11-10.md) for complete metrics and timings.

Complete g(0,0) has ~8.6e-5m/s² bias; subtraction cancels it at zero wind.
Positive-work penalty is soft, not a proof of universal dissipation; a small
positive residual was observed in1/600 zero-omega pair arms. No multi-seed,
hidden-parameter sensitivity, GPU consistency or measured-aircraft validation.

## Runtime selection and validation

```bash
python scripts/run_simulation.py --wind-backend pinn-x4 \
  --reference-config outputs/pinn/x4_950_wind_v1/reference_config.json \
  --pinn-checkpoint outputs/pinn/x4_950_pilot_v1/training_corrected/candidate.pt
```

The above explicit original-artifact selection still works. Ordinary default
startup is `python scripts/run_simulation.py`, using bundled paths independent
of the current working directory. Explicit fallbacks need only
`--wind-backend analytic` or `--wind-backend pinn`; the latter loads no new
checkpoint/reference. Overrides remain available, with no silent fallback.

Adapter checks kind/features/normalization against loaded tensors, units,
reference content/hash, mass, arm pose, XML/mesh closure and compiled-model
mass/initial pose/fluid state. CPU one Torch thread; old RPM/normalizer unused.
Logs identify backend, checkpoint/data/reference fingerprints and source code.
Keep source/weight licensing review separate before publication.

Runtime offline check: full suite220passed; actual-candidate500-row app scan
reproduces1.282% error, rawZ retained, actualFz0, force pipeline error0.
After the authorized default switch, the full offline suite is221passed,
including a no-argument app loading the bundled checkpoint. No new SITL round
was run for this selection/path-only change; earlier evidence is retained.
Real candidate/analytic SITL:24/24 clean complete runs, hoverOFF/+X1/2/3m/s,
three restarts each. All observed Fz exactly0, main-window clip/outside-domain
fractions0, all12 paired substitution gates pass (20mm absolute term dominates;
not a claim of relative differences<=10%). Both complete wind paths have P95
around1.2ms; no closed-loop wind-path speed gain established. No other
directions/trajectories or interactive GUI acceptance is implied. See the
[integration report](../X4_PINN_INTEGRATION_2026-11-10.md) for raw metrics/limits.
