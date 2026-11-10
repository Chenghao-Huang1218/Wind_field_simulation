# Local PX4 runtime

`bin/` and `etc/` contain the Linux x86-64 SITL rebuilt on 2026-11-10 from the
project's MuJoCo/X4-950 source. This verified runtime is included in the first
GitHub version; the large PX4 source tree and build cache are omitted from that
version. Existing local sources are preserved separately.

Run `bash scripts/run_px4_sitl.sh` from the project root. The host must provide
the linked Gazebo Transport13, Gazebo Messages10, Protobuf and C/C++ libraries;
check with `ldd runtime/px4/bin/px4`. This is a host SITL program and startup
resources. System libraries are not bundled, so arbitrary hosts are unverified.

Binary SHA256:
`db1c803bb28077856f8989b059c12421e0c8d3815be69aa3bb81b3262ae735ca`.
Source/patch/dependency records remain in `docs/px4.md` and `docs/provenance/`.
The same binary completed the recorded headless SITL comparisons; see
`docs/X4_PINN_INTEGRATION_2026-11-10.md`. Rebuilding requires the matching full
source checkout, which is not included in this runtime-only first version.
