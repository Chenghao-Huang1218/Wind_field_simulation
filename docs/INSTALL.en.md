# Installation and operation (English)

[Project homepage](../README.en.md) | [简体中文](INSTALL.zh-CN.md)

This guide targets **Ubuntu 24.04 Desktop, amd64 / x86-64, Bash**. The repository
includes the PX4 SITL runtime, models and both generations of PINN weights.
Normal use requires neither compiling PX4 nor retraining a model. Other Linux
versions, ARM, Windows and macOS have not been validated with this runtime.

## Dependency versions and validation boundaries

| Component | Observed tested version / requirement |
| --- | --- |
| OS / architecture | Ubuntu 24.04.5 LTS (Noble) / x86-64 |
| glibc / C++ runtime | glibc 2.39; host libstdc++6 14.2.0 |
| Python | Conda 3.12.13 on the tested host; Ubuntu Python 3.12 for new installs |
| MuJoCo | `3.6.0` |
| NumPy | `2.4.4` |
| PyTorch | `2.11.0`; tested host has `2.11.0+cu128`, with training and inference on CPU |
| Matplotlib | `3.10.9` |
| Tk | 8.6 on the tested host; the panel needs an interactive plotting backend |
| Gazebo Transport | Library `13.6.0`, ABI `libgz-transport13.so.13` |
| Gazebo Messages | Library `10.4.0`, ABI `libgz-msgs10.so.10` |
| ROS Jazzy vendor packages | `gz-transport-vendor 0.0.9`, `gz-msgs-vendor 0.0.8`; transitive dependencies include gz-utils2 |
| Protobuf | Host `3.21.12`, Ubuntu package `libprotobuf32t64`, ABI `libprotobuf.so.32` |
| Other declared dependencies | `lxml>=4.9`, `pyserial>=3.5` |
| Offline validation | `pytest==9.1.1`; packaging tool `build>=1.2` |

Python dependency declarations are maintained in [pyproject.toml](../pyproject.toml),
[requirements.txt](../requirements.txt) and [requirements-dev.txt](../requirements-dev.txt).
`lxml` and `pyserial` are declared compatibility dependencies; they are absent
from the original `env_uav`, so their ranges are not exact tested versions.
Host versions are a reference snapshot, not a requirement to freeze system security updates.

The installation procedure below is based on official repository setup and the
binary's observed ELF dependencies. It has not been validated end to end in a
fresh VM. Existing offline and SITL results apply to the tested host; check
dependencies and validate operation again on a new machine. CUDA, an NVIDIA GPU
and a full ROS Desktop installation are unnecessary. ROS packages supply matching
Gazebo libraries here; MuJoCo remains the physics engine and MAVLink the control link.

## 1. Install system dependencies

Confirm OS and architecture:

```bash
cat /etc/os-release
uname -m
```

Run on Ubuntu 24.04 x86-64 with sudo and network access. Skip the ROS repository
configuration if it is already configured.

```bash
sudo apt update
sudo apt install -y software-properties-common curl ca-certificates git python3
sudo add-apt-repository -y universe

# Configure signing keys and the repository using the official ROS package.
WIND_ROS_APT_VERSION=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
curl -fL -o /tmp/wind-ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${WIND_ROS_APT_VERSION}/ros2-apt-source_${WIND_ROS_APT_VERSION}.noble_all.deb"
sudo dpkg -i /tmp/wind-ros2-apt-source.deb

sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip python3-tk \
  libgl1 libglfw3 libxkbcommon-x11-0 libstdc++6 libprotobuf32t64 \
  ros-jazzy-gz-transport-vendor ros-jazzy-gz-msgs-vendor
```

Gazebo Harmonic uses Transport 13 / Messages 10. This recipe uses the standard
Jazzy vendor directories under `/opt/ros/jazzy/opt/`, matching the tested host.
The full Gazebo simulator and running ROS nodes are unnecessary. Sources:
[ROS Jazzy installation](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html),
[official repository setup source](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Installation/_Apt-Repositories.rst),
[Gazebo Harmonic library versions](https://gazebosim.org/docs/harmonic/install/).

## 2. Clone and create the Python environment

These commands place the repository in your home directory. If already cloned,
enter the existing checkout instead of cloning again.

```bash
cd ~
git clone https://github.com/Chenghao-Huang1218/Wind_field_simulation.git
cd ~/Wind_field_simulation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU wheels suffice for simulation and pilot reproduction.
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pip check
```

PyTorch sources: [official installation guide](https://pytorch.org/get-started/locally/)
and [official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/).
`pymavlink` and the legacy model's required RotorPy parameters are bundled;
installing the full RotorPy package is unnecessary. Existing users may continue
using Conda `env_uav`; `.venv` is the independent setup for new readers.

## 3. Preflight environment checks

```bash
cd ~/Wind_field_simulation
source .venv/bin/activate
python -c 'import sys,mujoco,numpy,torch,matplotlib; print(sys.version); print("mujoco",mujoco.__version__,"numpy",numpy.__version__,"torch",torch.__version__,"matplotlib",matplotlib.__version__)'
ldd runtime/px4/bin/px4
runtime/px4/bin/px4 -h
python scripts/run_simulation.py --help
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg python -m pytest -q
```

No `ldd` entry may say `not found`. `px4 -h` checks executable loading without
starting a flight. The runtime-only checkout explicitly skips the single C++
header contract test when PX4 sources are absent. The initial publication check
was **220 passed, 1 skipped**; the full source checkout had **221 passed**.
Offline tests do not start real SITL and are not flight validation.

Interactive use needs a desktop display session and working OpenGL.
`MPLBACKEND=Agg` is only for offline tests; do not use it for the interactive
commands below. `--no-viewer` still shows the panel and is not a headless mode.

## 4. Complete two-terminal launch commands

**Terminal 1: PX4 SITL**

```bash
cd ~/Wind_field_simulation
source .venv/bin/activate
bash scripts/run_px4_sitl.sh
```

Keep terminal 1 running while it waits for the simulator connection.

**Terminal 2: MuJoCo and control panel**

```bash
cd ~/Wind_field_simulation
source .venv/bin/activate
python scripts/run_simulation.py
```

The default is the bundled new `pinn-x4`. For an explicit fallback, replace only
the last command in terminal 2:

```bash
python scripts/run_simulation.py --wind-backend analytic
# Or the legacy PINN:
python scripts/run_simulation.py --wind-backend pinn
```

Wait for `PX4: connected` → `Unlock` → confirmed `armed` → `Takeoff` → stable hover.
For wind, enter world X/Y components, click `Apply Wind`, wait for acceptance,
then click `Wind OFF` to enable it. Require `sqrt(X²+Y²) <= 3 m/s` and Z=0:
`[2,2,0]` is valid; `[3,3,0]` is invalid. Finish with `Land`, then `Exit` after
touchdown. Exit manages only the SITL child created by that launcher.
See the [README](../README.en.md) for panel details and training reproduction, and
the [new PINN integration/SITL report](X4_PINN_INTEGRATION_2026-11-10.md) for comparisons.

## Troubleshooting

- **Missing `libgz-*`**: check Jazzy vendor packages and library major versions.
  The executable records standard `/opt/ros/jazzy/opt/*_vendor/lib` search paths.
  For nonstandard locations, configure `LD_LIBRARY_PATH` appropriately and recheck
  all transitive dependencies with `ldd`.
- **Missing `libprotobuf.so.32` or incompatible glibc**: confirm Ubuntu 24.04 amd64.
  Do not use symlinks to disguise incompatible library ABIs. Other platforms
  require matching source and a rebuild; see [PX4 documentation](px4.md).
- **No control panel / noninteractive backend**: run in a desktop terminal,
  check `echo "$DISPLAY"` and remove test-only `MPLBACKEND=Agg`. With system Python,
  a `.venv` and `python3-tk`, try `MPLBACKEND=TkAgg python scripts/run_simulation.py`.
  Conda needs Tk in its own environment.
- **Ports occupied**: defaults are TCP 4560 and Offboard UDP 14580. Stop your
  earlier instance with `Ctrl+C` in its terminal; do not kill other projects by process name.
- **Different checkout location**: replace `~/Wind_field_simulation` with your
  actual path. Both terminals must use the same project/environment. If setting
  `PX4_STATE_DIR`, use the same value in both terminals.

Public visibility does not assign a single open-source license to all materials.
Review [third-party notices](../THIRD_PARTY_NOTICES.md) before reuse or redistribution.
