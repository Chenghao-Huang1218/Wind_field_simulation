# Optional X4 PINN backend implementation plan

> Execute inline with superpowers:executing-plans and TDD; one fresh final review.

**Goal:** Add the trained pilot as optional `pinn-x4`, offline verify, then run
real PX4 SITL A/B against the analytic reference, without PX4 edits.

**Spec:** Existing accepted X4 PINN design and retraining plan Tasks 5–6;
user explicitly confirmed optional integration, preserved XY/filter/limits,
offline first, then SITL. No new training or default switch.

**Architecture:** Runtime adapter validates the frozen bundle/compiled model
and checkpoint metadata, then evaluates the 6D body pair on CPU float32.
Return raw three-axis W increment for diagnostics; keep application projection,
EMA=.05, per-axis limit=6 m/s², mass conversion and root force-only unchanged.
`pinn` remains legacy default. `analytic` remains available. Bad candidate
metadata fails before bridge creation; fallback is explicit selection, never
silently relabel a run. No candidate copying over old assets.

## Tasks and checks

1. Adapter: new `X4PinnWindResidualModel(checkpoint_path, reference_path,
   xml_path, model)` in wind_backends.py. Reuse bundle/runtime checks. Validate
   kind/features/coordinates/mass/arm/reference/XML-mesh fingerprints. Reject
   invalid domain/state/nonfinite output. Full raw increment and exact w0.
   Extend predict_increment with optional project_xy=False, default unchanged.
   RED→GREEN tests: strict scalar interface, checkpoint mismatches, full raw Z,
   zero wind and domain rejection. No model retraining required.
2. CLI/app/logging: `--wind-backend pinn-x4 --pinn-checkpoint PATH
   --reference-config PATH`. Required paths scoped to the selected backend.
   Log checkpoint/data/reference fingerprints, device/threads and source code.
   RED→GREEN tests: legacy default/fallback, no unwanted loader, fail before
   bridge, Apply gating, exact XY EMA/clip/mass/force-only, frozen arm geometry.
   Run actual pilot holdout adapter scan then complete offline suite before
   any SITL launch. Store raw full vs applied XY separately.
3. Real SITL: separate output/state directories; copied immutable parameter
   seed, official instance6 ports4566/14586/sysid7, only owned-session shutdown.
   First matrix: hover, OFF/+X1/2/3 m/s × analytic/pinn-x4 ×3 restarts=24 runs.
   Seed7 order; pair backends per condition, alternate order. 70s observation,
   5s trim +60s primary window, enable wind after10s flying; fixed qpos0 teacher.
   Stop on failed flight, retain excluded probes. No controller parameter edits.
   Report position/height, attitude/actuator/clip, raw/applied force, joints,
   inference latency and outside-domain observations. Gate RMSE difference
   <=max(.1*RMSE_analytic,.02m), no new abnormal exits; report scope limits.
4. Independent final read-only review; fix any material finding once with
   RED→GREEN, repeat suite when code changes. Update README/handoff/model card
   and reports. Preserve old checkpoints/reference/PX4 fingerprints.

## Review focus

- Forged/inconsistent checkpoint physical metadata or normalization buffers.
- No unintended legacy construction on new/analytic selection.
- Raw Z diagnostics never enter applied force or EMA state.
- Zero/domain/NaN behavior is identical offline and in actual app calls.
- SITL evidence really uses selected backend, same parameters and complete
  windows; preserve failures and distinguish pilot fitting from physical truth.

## Ledger

- Setup: standalone user-owned project has no Git metadata; preserve in-place
  changes with source snapshots in this run's own output directory. No migration
  or new Git worktree applies. Existing design and user choice authorize scope.
- Task1–2: complete. Added adapter, explicit CLI and provenance logging.
  New tests observed RED on missing adapter then GREEN (35 adapter/legacy/core
  cases). Tight unit-quaternion rejection observed RED then GREEN; added rtol=0.
  Full offline suite: 220 passed in28.69s, no skips.
- Actual pilot/application offline scan complete: 500 heldout rows, XY RMSE
  .000655/.000617 m/s², normalized1.282%; rawZ max.0734 m/s² retained only in
  diagnostics; actualFz0, zero-wind raw0 across3000 states; force-chain error0.
  Apply zero wind retains existing XY EMA decay; switching OFF resets it.
- Existing _apply_wind_force, ResidualSafetyFilter and external-force function
  ASTs exactly unchanged from this task's source snapshot.
- Task3: protocol/order frozen; instance6 ports verified free, matrix launched
  only after offline checks. Real sockets require sandbox escalation (normal
  sandbox disallows sockets); explicitly authorized local experiment, no PX4
  code/config writes. All PX4 states under this run's new directory.
- Final independent read-only review requested on fresh context, model
  gpt-6-astra/high per executing-plans skill. Documentation/flight results pending.
- Final code review complete: no Critical/Important or source-change Minor.
  Independent runtime tests9passed5.45s. Reviewer requests explicit final owned
  launcher returncode/managed shutdown checks; summary script enforces these
  in addition to flight complete/cleanlog (status alone predates cleanup).
- Task3 ongoing: real matrix process unified-exec session94337; run outputs
  outputs/closed_loop/x4_pinn_compare_2026-11-10. Five24cases complete at this
  entry, no failures; case script contains no training/PX4-source writes.
  run_matrix.py resumes by skipping complete results, refuses failed cases.
  summarize_matrix.py requires all24, computes full metrics/same-state teacher
  shadow/per-repeat substitution gate; plot_results.py plots final CSV only.
- Documentation added: X4_PINN_INTEGRATION_2026-11-10.md (flight collection
  pending), model card models/x4-950-pinn-pilot-v1.md, README/oldcomparison flags.
  Finish by validating full24, compact provenance/hashes, plotting/visual QA,
  replacing pending result text and handoff latest status. No default switch.
- Midpoint: 12/24 real runs complete, all statuscomplete/launcher0/Fz0/cleanlogs;
  repeat1 all4conditions done, repeat2 conditions3&1 done (then2&0). Source
  remained frozen, integrity check passes. Matrix unified-exec session94337
  still running. Final 24-run summary must be computed, no early success claim.
- Task3 complete:24/24 real runs, no exclusions/failures, all60s primary
  windows complete, all launchers0/managed shutdowntrue/cleanfooter.
  All12 paired substitution gates pass; max positiveXY RMSE difference .847mm
  (20mm absolute gate dominates, not a <=10% relative claim). ObservedFz0,
  primary clip/outside/rotor-rail fractions0. Learned core AST unchanged.
- Final source/integrity verification: candidate/legacy weights, frozen
  reference, protected PX4/runtime files and original default parameters all
  unchanged; all12 runtime-source fingerprints consistent across24runs.
  Complete wind-pathP95 ~1.2ms both backends; no runtime speed gain established.
- Final report/table/plot/model card/handoff/README complete; original source
  snapshots and frozen run sources archived. Compact evidence under
  docs/provenance/x4-pinn-sitl-2026-11-10, fullraw under outputs/closed_loop/
  x4_pinn_compare_2026-11-10. Figure QA checked color/gray and layout, fixed
  marker clipping in latency panel, exported300DPI PNG/PDF/SVG.
- Review dispositions: parent verified completed flights/docs, as reviewer
  requested. GUI/other directions/trajectories/state extrapolation/real
  aerodynamic accuracy/retraining remain outside this matrix; no such
  evidence claimed. No deferred minor findings, no extra source fix pass.
- Scope choice: optional-only per user; leave legacy default unchanged even
  after first-matrix acceptance. No Git integration applies to this standalone
  non-Git workspace; retain records/snapshots rather than deleting them.
