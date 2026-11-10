# Independent pilot review and resolution

Read-only fresh-context reviewer: gpt-6-astra, high reasoning effort.
Reviewed sampling/dataset/model/training/evaluation modules, CLI scripts,
config, tests and saved artifacts against the accepted spec and pilot plan.
No subreviewers. No critical findings.

Important: normalizing acceleration per axis before a·v can reverse physical
power sign. Reproduced with two tests (positive and negative physical work,
unequal scales), both failed before correction. Fixed by calculating a·v
first and dividing by one positive scalar and velocity scale. Tests then
passed. Repeated training with same seed/data/config; corrected artifacts
are recommended, initial run retained. State tensors happen to be bitwise
identical; the penalty was inactive during this particular training run.

Minor: rename evaluation's applied-Z claim to projected-Z, because pilot
does not apply forces in SITL. Renamed; no deferred minor findings.

Reviewer's independently reproduced initial holdout normalized error 1.28193%,
matching saved axis RMSE, data counts and provenance. Corrected rerun gives
the same errors. Final tests after fix: 211 passed in 26.67s.

Boundaries considered but not judged as implemented capabilities:

- Real aerodynamic accuracy: no measurements, explicitly unclaimed.
- Candidate SITL/end-to-end timing: not integrated, explicitly unclaimed.
- Formal120k, sensitivity and multi-seed: not authorized for this pilot.
- Exact rest equilibrium/global dissipation: not guaranteed; observed full
  output bias and small positive-work residual documented in the report.
- Complete historical unrelated-file diff: no Git metadata; 184 protected
  hashes checked, and pilot paths do not write PX4/runtime files.
- Parent completed final documentation/plot QA/full-suite verification after
  review, without requesting a second review.

Disposition: one correctness fix pass, RED→GREEN and fresh full suite; candidate
remains separate. Results are synthetic-reference pilot diagnostics only.
