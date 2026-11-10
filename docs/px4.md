# PX4 source and runtime

## First GitHub version: prebuilt runtime

The first public version includes `runtime/px4/bin/` and `etc/` so compatible
Linux x86-64 hosts can run the verified SITL without compiling PX4 first.
It omits `third_party/PX4-Autopilot/`, build caches and old run outputs. The local
complete development tree is retained. Source baselines, the adaptation patch,
licenses and runtime hashes remain under `docs/provenance/` and `licenses/`.
The source/build sections below describe that full development checkout;
obtain the matching sources separately before using those build commands.

The executable dynamically links Gazebo Transport13, Gazebo Messages10,
Protobuf and system C/C++ libraries. `ldd runtime/px4/bin/px4` must resolve its
dependencies on the target host. Existing checked host paths are recorded in
`provenance/sitl-minimal-ldd.txt`; universal binary portability is not claimed.

`third_party/PX4-Autopilot/` contains PX4 source and its recursive dependency
sources, with only the local adaptations required by the MuJoCo/X4-950 bridge.
It is independent of the original project. The required MuJoCo models live in
the Python package; `dev_huang/X4-950.params` retains the vehicle parameter export.
Official PX4 modules/resources are retained; this is a cleanup of local changes,
not a replacement or broad reduction of the official flight stack.

## Provenance and local Git metadata

The upstream base is `c2fb48990aacebc938c4e9998c7d5348f858fd22` in the official
PX4 repository. `d421b08e425b8fec06093fb7025a648df665db26` is a later **local
archive commit**, not an official upstream revision. Its migration description
was `v1.17.0-alpha1-195-gd421b08e42-dirty`.

The original `snapshot.json`, `px4-working-tree.patch`, `px4-submodules.txt` and
`px4-source-sha256.txt` under `docs/provenance/` remain historical migration
records. In particular, the old working-tree patch excludes already committed
local adaptations and is not a complete official-to-current patch.

The current, independently applicable official-to-current delta is
[`provenance/px4-minimal-mujoco.patch`](provenance/px4-minimal-mujoco.patch).
Its base revision and seven payload-file SHA-256 values are recorded in
[`provenance/px4-minimal-mujoco.json`](provenance/px4-minimal-mujoco.json).
`px4-minimal-submodules.txt` records current upstream dependency revisions.

## Retained simulation adaptations

- `SimulatorMavlink.cpp/.hpp`: publish MuJoCo truth to controller-facing
  position, attitude and angular-velocity topics; read allocator outputs from
  `actuator_outputs_sim` consistently with the subscription.
- `LocalPositionFeedback.hpp` and its receiver: pair float `LOCAL_POSITION_NED`
  with `HIL_STATE_QUATERNION` by sender and timestamp, reject invalid/stale or
  reused pairs, preserve the same origin as Offboard setpoints. The launcher
  enables this with `PX4_SIM_FLOAT_STATE=1`, which requires `EKF2_EN=0`.
- `10016_none_iris`, airframe CMake registration and `x4_950_params.defaults`:
  load X4-950 configuration and preserve Motor1..4 output mappings. The raw
  X4-950 parameter export is retained for provenance.

PX4 position/attitude/rate controllers and control allocation use the upstream
implementation. State estimation is bypassed in the current simulation; this
does not validate EKF2 or a complete real-sensor estimation chain.

Removed local features: the `MPCWRENCH` debug-array thrust/torque input, enlarged
command-ACK queue and its test, EKF2 EV-yaw/magnetic-reset fix and its test,
custom 4022/4023 Gazebo airframes and RC-L1/X4-950 Gazebo model additions.
Unrelated copied development documents/generated test launchers were removed;
official repository support files were restored. Official Gazebo modules and
models were preserved for source/build compatibility.

PX4's Makefile and CMake require Git metadata. `prepare_px4_snapshot.py` creates
independent local snapshot repositories for the copied submodules and PX4 itself.
The generated commits represent copied working files, not upstream commits. The
top-level synthetic version tag is `v1.17.0-0.1.0-dev`; original upstream commit
IDs remain in the provenance files. No credentials, remote configurations or Git
object alternates are copied. Run this once if Git metadata has been omitted
when distributing a source archive:

```bash
python scripts/prepare_px4_snapshot.py
```

## Build

Use a PX4-compatible compiler, CMake, Ninja, and the dependencies in the copied
`Tools/setup/requirements.txt`. Those firmware build dependencies are separate
from the Python simulation requirements. The source tree's setup scripts are
provided for reference; this migration did not install system packages.

```bash
bash scripts/build_px4_sitl.sh
```

This compiles the `px4` executable target under the unchanged
`px4_sitl_default` configuration and does not launch it. Building the default
`all` target additionally builds unrelated Gazebo plugins and may download
OpticalFlow; the wrapper intentionally avoids that target. The wrapper sets
`GIT_SUBMODULES_ARE_EVIL=1` to keep the included submodule files and avoid automatic
submodule downloads or replacement of the snapshot. Some official external
build dependencies, such as Micro-CDR v2.0.1 used by XRCE-DDS, may still need
network access or a local cache. Python firmware-build dependencies are separate
from `env_uav`; on this host the Conda base Python has the build tools. For an
explicit interpreter and bounded parallelism:

```bash
bash scripts/build_px4_sitl.sh PYTHON_EXECUTABLE=/home/raids/miniconda3/bin/python -j4
```

Run a rebuilt binary using:

```bash
PX4_RUNTIME_DIR="$PWD/third_party/PX4-Autopilot/build/px4_sitl_default" \
    bash scripts/run_px4_sitl.sh
```

The current startup scripts already load `x4_950_params.defaults` and restore the
four allocator motor outputs. The Python parameter export is used to read
`MPC_THR_HOVER`; it is not automatically imported into PX4 by the Python app.

## Current runtime

On 2026-11-10, `runtime/px4/bin/` and `runtime/px4/etc/` were replaced with the
locally rebuilt minimal-adaptation executable and its generated startup files.
The existing launcher commands, active startup scripts, X4-950 defaults and
`outputs/px4/` parameters/state were preserved. The former migrated runtime was
backed up locally at `/tmp/wind-px4-cleanup-4co2a_ju/runtime/`; this temporary
backup is not a release artifact. Binary hash/build provenance is in
`provenance/px4-minimal-mujoco.json`, with runtime file checksums in
`provenance/px4-minimal-runtime-sha256.txt`.

The executable is a Linux x86-64 host build. Its current `ldd` results are in
`provenance/sitl-minimal-ldd.txt` (`sitl-ldd.txt` describes the migrated binary).
It still links to official Gazebo Transport 13 and
Gazebo Messages 10 from the host ROS installation. System libraries, GPU/OpenGL
drivers, the Python interpreter and general Python packages are not bundled.
On a host with incompatible dependencies, obtain the matching source checkout
and rebuild. The first GitHub version now tracks the verified `bin/` and `etc/`;
other generated runtime files stay ignored.

The cleanup build completed successfully. The rebuilt executable passed dynamic
library resolution and `px4 -h`, without starting a SITL session. A direct C++
test exercises the retained float-sample pairing helper; the full offline Python
suite also covers the existing application and EXIT supervisor. These checks
do not demonstrate flight performance or real panel-to-SITL EXIT integration.
The post-cleanup full suite passed **104 tests** (including the compiled C++
contract test, no skips). A content audit checked all 13,154 upstream tracked
files: four existing files carry the necessary delta, and three files are added
(receiver helper, generated X4-950 defaults, raw parameter export). The Gazebo
submodule matches its 194 upstream files with local model additions removed.

This host's build reused upstream Micro-CDR v2.0.1 sources through an isolated
project-local cache in `outputs/build-deps/`, with a process-local Git URL
redirect. It did not modify or retain a runtime dependency on the old project.
The cache's upstream revision is recorded in the build manifest. A fresh machine
must provide official build dependencies or network access; the source patch
alone does not bundle every external build dependency.

## Coordinated panel EXIT

`run_px4_sitl.sh` now delegates to the standard-library Python supervisor
`scripts/run_px4_sitl.py`. It owns exactly one PX4 child process, holds a file
lock in the state directory, and publishes a per-launch session identifier.
After the panel's EXIT cleanup completes, the simulation writes a shutdown
request for that session. The supervisor checks it at 0.1 s intervals and sends
SIGINT to its own child, using PX4's existing POSIX shutdown handler. If the
child does not exit within 10 s, it receives SIGTERM, followed after 5 s by a
final stop of that same child. Stored PIDs are informational and are never used
by the simulation to signal another process.

The default shared state directory is `outputs/px4/`. If `PX4_STATE_DIR` is
customized, export the same absolute path in both terminals. Stale session
requests are ignored, and a second launcher cannot use a locked state directory.
Terminal windows remain open when their programs finish.

An already running SITL started with the old launcher has no supervisor. Stop
that instance manually once and restart it with the updated launcher before
using coordinated EXIT. A missing manager produces a message in terminal two;
the simulation does not terminate PX4 processes by name.

At the EXIT fix, offline validation was 103 tests, including a subprocess lifecycle
exercise with a harmless fake SITL, session isolation, stale-record handling,
and the EXIT-to-supervisor request. No actual PX4 or flight simulation was
started for that fix; firmware source and the binary were unchanged at that
stage. The later source cleanup/rebuild above is a separate change.
