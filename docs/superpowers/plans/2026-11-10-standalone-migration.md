# Wind field simulation migration plan

Goal: Copy the current MuJoCo/PX4 wind simulation into an independent project.
Scope: Preserve numerical, UI, control, and MAVLink behavior; change resource
paths and import locations only. The user explicitly requested full PX4 sources.

- [x] Copy the application, recursive model assets, PINN dependencies and weights.
- [x] Add installable src layout, CLI, launch scripts and dependency requirements.
- [x] Copy complete PX4 source and initialized submodule contents, excluding Git
  history, build caches, old experiment data and unrelated local controller apps.
- [x] Preserve the existing SITL executable/startup resources and provenance.
- [x] Run migrated offline tests, isolated package/resource loading and relocation
  checks; do not start SITL or an interactive flight simulation.

Review focus: XML include/mesh paths, checkpoint compatibility, package assets in
wheels, external symlinks, and Git metadata required by PX4's build system.

All tasks verified; details and limits are recorded in docs/migration.md.
