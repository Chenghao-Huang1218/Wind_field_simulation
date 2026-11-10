# AM-Bench 柜子：MuJoCo 模型

由 AM-Bench 的 `cabinet.usd` 转换。保留柜体、隔板、门板、把手和原始贴图；所有几何使用 MuJoCo 原生 box/cylinder，无需 USD 或网格库即可运行。

## 启动（env_uav）

```powershell
conda activate env_uav
cd 'E:\博士科研\空中机械手\ambench-main\cabinet model'
python view_cabinet.py
```

自动演示滑动柜门的开关：

```powershell
python view_cabinet.py --demo
```

演示每 4 秒交替施加 ±8 N 的关节力；柜门通过动力学运动，没有直接改写位置。关闭窗口退出。普通模式不施加驱动力，可供机器人通过接触操作。

也可以直接使用环境解释器：

```powershell
& 'D:\Anaconda\envs\env_uav\python.exe' 'E:\博士科研\空中机械手\ambench-main\cabinet model\view_cabinet.py' --demo
```

## 文件

- `cabinet.xml`：柜子本体，含 36 个去重后的基本几何、1 个滑动关节、材质和接触排除规则。
- `scene.xml`：可运行场景，引用柜子并增加地面、灯光。
- `textures/`：从源模型复制的木材和金属贴图。
- `cabinet_loader.py`：通过 MuJoCo 内存资产接口加载，兼容 Windows 中文路径。
- `view_cabinet.py`：交互查看和外力开关门演示。
- `validate_cabinet.py` / `validation.json`：验证程序与本次结果。
- `preview_closed.png` / `preview_open.png`：MuJoCo 实际渲染。
- `conversion_manifest.json`：每个几何与原 USD 路径对应、关节参数和近似说明。
- `convert_cabinet.py`：专用于此柜子的可复现转换脚本；重跑转换才需要 `usd-core` 和原 USD 的关联资源。
- `LICENSE.source`：源项目许可证副本；模型和贴图的原有权利归属不变。

## 坐标与控制

单位为 m、kg、s，Z 朝上，保留 USD 原点。柜体正面朝 -X，主体大致在 x=0.6～1.22、y=-0.6～0.6、z=0～1.68 m。场景未叠加原任务的 (1.5, 0, 0) 环境平移。

目标关节名 `pull_door_slide`，机体名 `pulling_door`，把手参考点 `door_handle`。关节方向约为 +Y，范围 [-0.6, 0] m：0 为关闭，-0.6 为打开。对应 USD 的 joint-frame X 经 localRot1 旋转后的实际方向，并非世界 X。

`cabinet.xml` 默认无 actuator。对门施加接触力或 `mj_applyFT` 可驱动；演示为了独立运行，用等效关节力 `qfrc_applied`。模型提供 closed/open 两个 keyframe，供初始化使用。

在本目录运行自定义脚本时：

```python
import mujoco
from cabinet_loader import load_cabinet

model = load_cabinet()
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
```

整合到其他 MJCF 时可引用 `cabinet.xml`，但注意该文件也含 compiler、option、asset、contact 和 keyframe；与已有场景合并时应统一全局选项和 keyframe 的 qpos 维数，不能直接向多自由度场景沿用单自由度 keyframe。不要丢失 textures 目录。

## 保真范围与近似

1. 使用 OpenUSD 解析组合层和实例代理，读取实际变换、尺寸与关节参数。去除了 36 个与同一刚体几何完全重合的重复视觉形状，避免重叠渲染和重复质量。
2. 仅保留 cabinet_pick_place 任务使用的滑动门。其余抽屉和门板固定，包括源资产限位为 0 的铰链门，以及 jointEnabled=false 的其他部件；这不是通用的全部关节转换器。
3. 源资产的门体没有显式有效质量/惯量（mass=0、inertia=0 表示交由引擎计算）。这里明确用 1000 kg/m³ 和去重后的基本几何估算，门质量约 3.2353 kg。不是经过识别的真实门质量；科研实验应按实测值重新标定。
4. 从任务配置保留 damping=10、stiffness=0；将 friction=2 近似为 MuJoCo 的 frictionloss=2 N。这两个引擎的摩擦含义不完全相同。接触摩擦系数 0.7、扭转 0.005、滚动 0.0001 是转换模型选值，非源模型标定值。
5. 使用基本几何作为碰撞体，保留柜内空腔与把手间隙。门与固定柜体之间显式关闭接触，参照源关节 collisionEnabled=false；外部机器人/物体与门、柜体仍可碰撞。这是对柜体全部固定部件的统一排除。
6. 原木纹和金属图片已打包，但 USD 的 MDL 投影贴图与 MuJoCo 材质不同，光照/纹理投影仅近似。
7. 默认没有机器人、罐子、摄像头任务、奖励函数或控制器，仅包含柜子和独立查看场景。

## 验证

```powershell
python validate_cabinet.py
```

使用 env_uav 的 MuJoCo 3.6.0 验证：模型编译、单滑动自由度、0.6 m 行程、把手位移、静止稳定性、±8 N 外力开关门、外部探针与把手碰撞，以及开/关门图像渲染。关节限位是软约束，8 N 持续载荷下约有 0.11 mm 越限；没有数值警告。

这证明模型可加载并完成基础动力学运行，不代表它与 Isaac Lab 的接触力和开门阻力已完成定量对齐。

## 重现转换

另外安装 usd-core（运行模型无需安装）：

```powershell
python convert_cabinet.py '原始cabinet.usd完整路径' '输出文件夹'
```

转换脚本读取同级 configuration/materials 贴图目录；输出目标中的同名模型和贴图会更新。

参考：[MuJoCo MJCF 文档](https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html)。
