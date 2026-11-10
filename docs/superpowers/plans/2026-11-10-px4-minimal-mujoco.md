# Minimal MuJoCo PX4 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution. The user has explicitly authorized this cleanup after reviewing the retained and removed features in chat.

**Goal:** Retain only the local PX4 changes needed by the existing MuJoCo/X4-950 simulation.

**Architecture:** Keep simulator truth feedback, paired float NED input, allocator motor output and the X4-950 defaults. Restore unrelated core changes to official baseline `c2fb48990aacebc938c4e9998c7d5348f858fd22`, and remove custom Gazebo airframes/models. Rebuild locally and promote the verified runtime without changing launcher commands or existing state.

**Tech Stack:** PX4 C++, CMake/Ninja, Python 3.12/pytest, POSIX shell.

**Spec:** User request in this chat on 2026-11-10: simplify project-local PX4 modifications, preserve the simulation method, never modify `/home/raids/PX4-Autopilot`.

## Global Constraints

- All mutations remain in this project or `/tmp`. The original repository is read-only reference material.
- Do not change the official controller algorithms, estimator architecture, Python bridge, ports, EXIT ownership or existing PX4 state.
- Retain BSD and nested notices. Historical migration records remain historical.
- Report source checks, compilation and offline tests separately from SITL flight/EXIT verification.
- This project has no top-level Git repository; use a targeted backup and file manifests, without initializing a project repository or inventing commits/worktrees.

## Design decisions

- Chosen approach: restore unrelated files to the exact official baseline and preserve only the existing required delta. Rewriting the bridge or switching to EKF2 changes the user's simulation method and is outside scope.
- Restore the MAVLink receiver, command ACK queue/test, and EKF2 magnetometer/vision test files to baseline.
- Remove custom Gazebo 4022/4023 airframes and four local Gazebo model directories, preserving official Gazebo code/resources to avoid gratuitous build-system changes.
- Remove unrelated copied development documents and generated POSIX test launchers; restore official repository support files and `.gitignore` so the remaining local source delta is explicit.
- Preserve X4-950 parameters, 10016 startup, airframe registration, SimulatorMavlink.cpp/.hpp and LocalPositionFeedback.hpp.
- Keep original runtime until a local rebuild succeeds. Back it up before replacing bin/etc. Preserve outputs/px4 unchanged.
- Add an independently applicable official-baseline patch and the added required files, with a reproducibility manifest. Preserve original provenance files.

## Review Focus

- No local thrust/torque command entry remains available in the promoted binary.
- Current simulator feedback and allocator output remain byte-identical to the pre-cleanup implementation.
- Required X4-950 defaults/motor mappings remain loaded; existing saved parameters are not overwritten.
- Submodule preparation neither fetches nor restores deleted historical content.
- Rebuild failure must leave the existing runtime usable; runtime promotion must preserve startup paths.

## Task 1: Remove historical source changes

- [x] Back up every modified/deleted file and runtime hashes under `/tmp`; capture an original-repository read-only fingerprint.
- [x] Add contract verification of the actual retained C++ sample pairing and required feedback/output configuration; include mismatch/age/invalid/single-use/origin cases.
- [x] Run the cleanup validation before cleanup and observe historical changes being rejected.
- [x] Restore unrelated core files and support files to official baseline; remove only identified local extras.
- [x] Produce retained-delta patch and current submodule provenance, leaving migration records intact.
- [x] Re-run contract and baseline/delta validation.

## Task 2: Build and deliver consistent runtime

- [x] Compile using project-local source and local dependencies; no downloads or source changes outside this project.
- [x] Check executable/library resolution, removed interface absence and required feedback presence.
- [x] Back up existing runtime and promote verified bin/etc with same-filesystem renames and rollback while preserving state.
- [x] Update README, PX4 documentation, handoff and runtime provenance to distinguish current minimal source/runtime from the migrated snapshot.
- [x] Run the full Python suite and review the full cleanup delta; record build and flight verification boundaries.

## Progress

- 2026-11-10: Plan created; no top-level Git repository. Native execution authorized by the direct implementation request. No flight run is required to claim source cleanup or build success.
- Task 1 complete: source backup `/tmp/wind-px4-cleanup-4co2a_ju`; cleanup audit failed before removal and passed afterwards; retained C++ contract passed. Baseline Python suite: 103 passed.
- Ruling: build the `px4` target under the unchanged `px4_sitl_default` configuration. The official `all` target includes Gazebo OpticalFlow ExternalProject downloads, which failed due to unavailable network and are not used by MuJoCo. This changes only the wrapper's build scope, not firmware configuration or control behavior.
- Ruling: use a project-local offline Micro-CDR v2.0.1 source cache, exported read-only from existing upstream Git objects, with a process-local URL redirect. The default PX4 configuration requires this external dependency; no source/configuration change was needed. A fresh machine still needs dependency sources/network access; this host-local cache is not a release payload.
- Task 2 complete: full compilation passed; ldd resolved dependencies; px4 -h exited 0; prior binary contains MPCWRENCH and new binary does not. Promoted bin/etc; 4 saved state files unchanged. Source/runtime backup is `/tmp/wind-px4-cleanup-4co2a_ju`.
- Verification: 104 tests passed in 8.48s with no skips. Official main-tree content audit: 13,154 files, exactly 4 intended modifications; Gazebo upstream content audit: 194 files unchanged. Minimal patch applies to official baseline and reproduces seven payload files.
- Final fresh-context source/build-target review: no important correctness findings. Runtime promotion and final documentation were checked by the implementer. No flight or real panel EXIT verification performed.
