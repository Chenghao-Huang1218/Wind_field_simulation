# Horizontal Wind Inputs

User request: remove Z wind input and allow only horizontal wind, preserving the successfully tested Apply-before-enable workflow.

- [x] Add failing UI/runtime/CLI coverage: two input fields, submitted `(X,Y,0)`, vertical setup rejected without losing accepted horizontal settings, `--wind-z` unavailable.
- [x] Remove Z widget and CLI flag, lock the setup vector's Z component to zero through shared validation, preserve the PINN's three-dimensional state/force interface.
- [x] Update existing wind tests and README/handoff; render the two-field panel and run the full offline suite.

Keep wind absent at startup; only accepted Apply unlocks the switch. Do not suppress vertical forces predicted by the PINN from horizontal airflow. No PX4, vehicle position-jog, flight or EXIT changes. User reports the preceding Apply gate works in actual simulation; this horizontal-only revision still needs separate interactive verification.

Verification: four new requirement cases failed before implementation; final full offline suite 132 passed in 16.32s with no skips. Visually inspected the rendered two-field panel, including accepted settings and unchanged position-jog controls. Python syntax and README commands passed validation. Backup: /tmp/wind-horizontal-before. No SITL session started.
