# Runtime Wind Settings Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for this bounded inline implementation.

**Goal:** Let the user change wind speed/direction in the running control panel without restarting either terminal.

**Architecture:** Add signed world-frame X/Y/Z velocity text boxes and an explicit Apply Wind button. Validate finite values as a complete vector, queue SET_WIND, then replace the immutable ConstantWindField on the physics loop. Keep CLI values as initial configuration and leave the wind enable flag and PINN/EMA force path unchanged.

**Tech Stack:** Python/NumPy, Matplotlib TextBox/Button, pytest with Agg.

**Spec:** User's direct request in this chat: set wind speed and direction on the panel during execution, avoiding CLI edits/restarts. The existing CLI and data interface already use world XYZ components in m/s; retaining that representation is the smallest compatible implementation.

## Global Constraints

- Modify only the independent Wind_field_simulation project. No PX4 or firmware changes.
- Keep the compact 7.4 × 5.0 inch panel; retain flight controls, diagnostic text and EXIT lifecycle.
- Applying settings must not implicitly enable wind, restart the simulation, alter targets/flight mode or reset the existing residual filter.
- Invalid/partial/non-finite input must leave all three accepted components unchanged and show an inline error.
- Repeated panel refreshes must not overwrite drafts, including edits made after submitting a prior update.
- Preserve type hints and Google Style docstrings; no new numerical randomness.

## Review Focus

- Initial CLI values are displayed correctly, including non-default values.
- Accepted runtime values reach PINN inputs, arrow state and snapshots rather than stale immutable CLI config.
- An invalid command delivered programmatically cannot crash the simulation or partially update settings.
- Panel widgets do not overlap existing controls or interfere with position-jog clicks.
- Settings changes while OFF remain disabled; while ON they affect subsequent force evaluation through the existing filter.

## Tasks

- [x] Write failing UI and runtime tests: atomic valid enqueue, invalid rejection, initial CLI values, draft preservation, runtime ON/OFF and snapshot/PINN consistency.
- [x] Add SET_WIND handling, runtime snapshot source and Matplotlib inputs/feedback within the current panel dimensions. Keep callbacks queue-only.
- [x] Run targeted tests, then render and inspect the panel (normal and invalid-input states), adjusting layout as needed.
- [x] Update README operating sequence and handoff for Apply Wind and CLI defaults.
- [x] Run the full offline suite and one fresh-context correctness review. Report separately from interactive SITL verification.

## Progress

- User has requested implementation; the XYZ/Apply approach reuses the established interface without requiring a new wind-direction convention. No top-level Git repository; work in place and preserve a targeted pre-change backup under /tmp.
- Backup: `/tmp/wind-runtime-settings-before/`. Fourteen new behavior cases failed before implementation (missing SET_WIND/editor) and passed afterwards. A subsequent actual mouse-click test verifies Apply Wind wiring without producing jog commands.
- Full offline suite: 119 passed in 13.85s, no skips. Rendered and visually inspected normal and invalid-input panels; no overlap in the new controls, feedback or Exit. No interactive SITL was started.
- Final fresh-context review found one important numerical issue: finite float64 wind values such as 1e20 and 1e40 overflow the real float32 PINN norm/drag path. Four reproducing cases failed before the fix and passed after it, including real CPU inference after rejection.
- Ruling: enforce a shared numerical envelope `sqrt(float32.max * float32.eps)` in the editor and ConstantWindField to reserve precision-factor headroom in squared speed. This rejects extreme finite vectors that previously could terminate an active session; it does not assert a trained or physically validated wind range.
- Final suite after the review fix: 123 passed in 14.59s, no skips. Source syntax and README startup/link checks passed. No further review findings or deferred minors. Interactive SITL wind adjustment remains unverified.
