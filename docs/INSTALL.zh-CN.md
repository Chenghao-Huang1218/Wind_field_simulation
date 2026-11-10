# 安装与运行（中文）

[项目主页](../README.md) | [English](INSTALL.en.md)

本指南面向 **Ubuntu 24.04 Desktop、amd64 / x86-64、Bash**。仓库已提供 PX4 SITL
运行包、模型和两代 PINN 权重，正常使用无需编译 PX4 或重新训练。
其它 Linux 版本、ARM、Windows、macOS 未验证；不可将本运行包当作这些平台的通用二进制。

## 依赖版本与验证边界

| 项目 | 本机实际验证版本／要求 |
| --- | --- |
| 操作系统 / CPU 架构 | Ubuntu 24.04.5 LTS（Noble）/ x86-64 |
| glibc / C++ 运行库 | glibc 2.39；本机 libstdc++6 14.2.0 |
| Python | 本机 Conda 3.12.13；新安装使用 Ubuntu Python 3.12 |
| MuJoCo | `3.6.0` |
| NumPy | `2.4.4` |
| PyTorch | `2.11.0`；本机为 `2.11.0+cu128`，训练和气动推理均使用 CPU |
| Matplotlib | `3.10.9` |
| Tk | 本机 8.6；控制面板需要交互式绘图后端 |
| Gazebo Transport | 库 `13.6.0`，ABI `libgz-transport13.so.13` |
| Gazebo Messages | 库 `10.4.0`，ABI `libgz-msgs10.so.10` |
| ROS Jazzy vendor 包 | `gz-transport-vendor 0.0.9`、`gz-msgs-vendor 0.0.8`，传递依赖含 gz-utils2 |
| Protobuf | 本机 `3.21.12`，Ubuntu 包 `libprotobuf32t64`，ABI `libprotobuf.so.32` |
| 其余声明依赖 | `lxml>=4.9`、`pyserial>=3.5` |
| 离线验证 | `pytest==9.1.1`；打包工具 `build>=1.2` |

Python 版本声明以 [pyproject.toml](../pyproject.toml)、[requirements.txt](../requirements.txt)
和 [requirements-dev.txt](../requirements-dev.txt) 为准。`lxml` 和 `pyserial` 是项目声明的
兼容依赖，本机原 `env_uav` 未安装它们，不能将其版本范围说成已测试的精确版本。
表中的宿主版本是参考快照，不要求冻结所有系统安全补丁。

以下安装步骤根据官方仓库配置和本机 ELF 依赖编写，尚未在全新虚拟机完成端到端安装验证。
既有离线测试和 SITL 结果对应本机已测环境；在新机器上需重新检查依赖并完成运行验证。
CUDA、NVIDIA GPU 和完整 ROS Desktop 均不是本项目的运行要求；ROS 软件源在这里用于
提供匹配的 Gazebo 动态库，物理仿真仍使用 MuJoCo，控制通信仍使用 MAVLink。

## 1. 安装系统依赖

先确认系统与架构：

```bash
cat /etc/os-release
uname -m
```

在 Ubuntu 24.04 x86-64 上执行。需要 sudo 和网络访问；已有 ROS 2 软件源时可跳过其配置段。

```bash
sudo apt update
sudo apt install -y software-properties-common curl ca-certificates git python3
sudo add-apt-repository -y universe

# 使用 ROS 官方 ros2-apt-source 包配置签名密钥及软件源。
WIND_ROS_APT_VERSION=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
curl -fL -o /tmp/wind-ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${WIND_ROS_APT_VERSION}/ros2-apt-source_${WIND_ROS_APT_VERSION}.noble_all.deb"
sudo dpkg -i /tmp/wind-ros2-apt-source.deb

sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip python3-tk \
  libgl1 libglfw3 libxkbcommon-x11-0 libstdc++6 libprotobuf32t64 \
  ros-jazzy-gz-transport-vendor ros-jazzy-gz-msgs-vendor
```

Gazebo Harmonic 对应 Transport 13 / Messages 10。这里采用与本机相同的 Jazzy vendor
目录 `/opt/ros/jazzy/opt/`，无需安装整个 Gazebo 模拟器，也无需启动 ROS 节点。
依据：[ROS Jazzy 安装说明](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)、
[官方软件源配置原文](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Installation/_Apt-Repositories.rst)、
[Gazebo Harmonic 库版本](https://gazebosim.org/docs/harmonic/install/)。

## 2. 获取项目并创建 Python 环境

下列命令默认将仓库放在用户主目录。若已克隆，请进入已有目录，不要重复克隆。

```bash
cd ~
git clone https://github.com/Chenghao-Huang1218/Wind_field_simulation.git
cd ~/Wind_field_simulation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU 版足以运行及复现 pilot，避免下载不需要的 CUDA 依赖。
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pip check
```

PyTorch CPU 安装来源：[官方安装说明](https://pytorch.org/get-started/locally/)、
[官方 CPU wheel 索引](https://download.pytorch.org/whl/cpu/torch/)。
项目携带 `pymavlink` 和旧模型所需的 RotorPy 参数，不需另装完整 RotorPy。
已有 Conda `env_uav` 可继续使用；上述 `.venv` 是给新读者的独立安装方式。

## 3. 启动前检查

```bash
cd ~/Wind_field_simulation
source .venv/bin/activate
python -c 'import sys,mujoco,numpy,torch,matplotlib; print(sys.version); print("mujoco",mujoco.__version__,"numpy",numpy.__version__,"torch",torch.__version__,"matplotlib",matplotlib.__version__)'
ldd runtime/px4/bin/px4
runtime/px4/bin/px4 -h
python scripts/run_simulation.py --help
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg python -m pytest -q
```

`ldd` 输出中不能有 `not found`。`px4 -h` 仅检查程序能否加载，不启动飞行。
运行包仓库缺少 PX4 源码时，单项依赖 C++ 头文件的契约测试会明确跳过；首版发布检查为
**220 passed, 1 skipped**，完整源码目录对应 **221 passed**。离线测试不会启动真实 SITL，
通过不等于完成飞行闭环验证。

手动使用需要桌面显示会话及可用 OpenGL。`MPLBACKEND=Agg` 仅用于离线测试，不要将它
用于下面的交互启动。`--no-viewer` 仍保留控制面板，不是无界面模式。

## 4. 两个终端的完整启动命令

**终端一：PX4 SITL**

```bash
cd ~/Wind_field_simulation
source .venv/bin/activate
bash scripts/run_px4_sitl.sh
```

保持终端一运行，它会等待仿真连接。

**终端二：MuJoCo 与控制面板**

```bash
cd ~/Wind_field_simulation
source .venv/bin/activate
python scripts/run_simulation.py
```

默认加载打包的新 `pinn-x4`。显式回退时，仅替换终端二最后一行：

```bash
python scripts/run_simulation.py --wind-backend analytic
# 或：旧 PINN
python scripts/run_simulation.py --wind-backend pinn
```

等待 `PX4: connected` → `Unlock` → 确认 `armed` → `Takeoff` → 稳定悬停。
需要风扰时输入世界系 X/Y 风速，点击 `Apply Wind`，被接受后点击 `Wind OFF` 开启。
要求 `sqrt(X²+Y²) <= 3 m/s`、Z=0；例如 `[2,2,0]` 有效，`[3,3,0]` 无效。
结束时 `Land`，落地后 `Exit`；退出只管理该启动器创建的 SITL 子进程。
详细面板说明和训练复现见 [README](../README.md)，对照流程见
[新 PINN 接入与 SITL 报告](X4_PINN_INTEGRATION_2026-11-10.md)。

## 常见问题

- **缺少 `libgz-*`**：核对 Jazzy vendor 包和对应主版本。二进制记录了标准
  `/opt/ros/jazzy/opt/*_vendor/lib` 搜索路径。非标准安装位置应配置自己的
  `LD_LIBRARY_PATH`，并重新运行 `ldd` 检查所有传递依赖。
- **缺少 `libprotobuf.so.32` 或 glibc 版本不兼容**：核对是否为 Ubuntu 24.04 amd64，
  不要用不同主版本库的软链接冒充 ABI；其它平台应另取匹配源码重编译，见 [PX4 说明](px4.md)。
- **控制面板不显示或提示非交互后端**：在桌面终端中运行，检查 `echo "$DISPLAY"`，
  去掉测试用的 `MPLBACKEND=Agg`。系统 Python 的 `.venv` 已安装 `python3-tk` 时，
  可用 `MPLBACKEND=TkAgg python scripts/run_simulation.py`。Conda 的 Tk 需在其环境中另行提供。
- **端口被占用**：默认 TCP 4560、Offboard UDP 14580；在旧实例自己的终端 `Ctrl+C`
  退出后重试，不按进程名批量终止其它项目。
- **运行与文档中的新机器路径不同**：将 `~/Wind_field_simulation` 替换为实际克隆位置，
  两个终端使用同一项目、同一 Python 环境；自定义 `PX4_STATE_DIR` 时两边保持相同。

公开可见不代表全部材料获得统一开源授权，使用及再分发前请查看
[第三方声明](../THIRD_PARTY_NOTICES.md)。
