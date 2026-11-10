# Require Apply Wind Before Enabling Wind

**Goal:** Start with no configured disturbance and require one accepted Apply Wind before Wind ON/OFF is usable.

**Approach:** Represent absent parameters explicitly as None in configuration/runtime/snapshots. Keep CLI wind options as optional draft-prefill compatibility only; they cannot configure or enable wind. Guard toggling in the flight state machine and disable the real panel widget until a simulation snapshot confirms accepted settings. Preserve valid zero-vector application, runtime edits and invalid-input rejection.

**Scope:** Project-local simulation.py, behavior tests, README/handoff. No PX4, model, filter or EXIT changes. Direct implementation requested by user; execute in place (no top-level Git). Backup: /tmp/wind-apply-gate-before.

- [x] Add failing tests for absent defaults, blank/disabled panel, blocked direct toggle, rejected Apply remaining locked, and valid/zero Apply enabling the actual widget only after acknowledgement.
- [x] Implement optional wind state, runtime and UI gates, and no-force behavior before configuration. Update old tests to explicitly apply their required wind fixtures.
- [x] Run full offline tests, inspect initial/accepted panel renders, and perform a fresh read-only review.
- [x] Update operating instructions and verification record.

Initial wind setup does not enable wind. Turning wind off retains the accepted setup, so subsequent toggles do not require another Apply. A later invalid draft retains any previously accepted setup.

Verification: six updated/new requirement cases failed before implementation; final full offline suite 128 passed in 17.65s with no skips. Updated old tests explicitly configure wind before exercising force or toggle behavior. Rendered and inspected unconfigured/disabled and accepted/enabled-switch panel states; startup commands and Python syntax verified. No real SITL session started.

Fresh read-only review passed with no correctness findings; reviewer also ran 35 focused wind/Apply cases successfully. No changes to PX4, physics/filter algorithms or EXIT lifecycle.
