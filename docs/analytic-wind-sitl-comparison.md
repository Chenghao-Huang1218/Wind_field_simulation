# 可选解析气动后端与 SITL 对照步骤

更新：2026-11-10。新pinn-x4现为默认，解析与旧PINN保留显式回退。
现已完成首轮真实无窗口 SITL 悬停 OFF/1 m/s +X 对照，见 [结果](SITL_FIRST_ROUND_2026-11-10.md)。
以下为 GUI 手动步骤，鼠标/窗口路径及其它风速/轨迹不由首轮结果代替验证。
不需要训练、编译或修改 PX4。

后续新 `pinn-x4` 已完成悬停OFF/+X1/2/3m/s与解析的24次真实对照，
结果与新旧选择命令见 [接入报告](X4_PINN_INTEGRATION_2026-11-10.md)。

## 后端与施力边界

| 选项 | 行为 |
| --- | --- |
| 无后端参数 / `--wind-backend pinn-x4` | 默认新6维X4网络，使用打包权重与冻结参考 |
| `--wind-backend pinn` | 显式回退原权重与normalizer，不加载新权重 |
| `--wind-backend analytic` | 显式回退同一打包解析参考，不加载网络 |
| `--run-log PATH` | 可选 50 Hz JSONL 对照记录；文件必须尚不存在 |

三个后端共用 `predict(v_O_world, quaternion_wxyz, omega_body, wind_world)`，
输出原始三轴世界系加速度；施力继续走 **XY 投影 → EMA(alpha=0.05) →
逐轴 ±6 m/s² 限幅 → 乘整机质量 → 根 body 单点 force-only** 链路。
原始诊断 Z 可非零，实际风力 Z 和气动力矩为零。
未启用 MuJoCo fluid，不额外施加无风完整阻力或分布式气动力矩。
关闭风扰清空原滤波；更新已接受风速保持 XY EMA、目标和开关状态。

CLI 草稿、面板和 SET_WIND 对三个后端均限制 **水平合风速 ≤3 m/s、Z=0**。
例如 `[3,0,0]`、`[0,-3,0]`、`[2,2,0]` 有效，`[3,3,0]` 超域。
预填仍不能绕过 Apply；非法提交保留上一组已接受设置。

解析启动校验 reference/vehicle/geometry 指纹、版本、质量/代理/关节一致性，
以及当前场景 include 闭包、XML/mesh 散列和编译模型质量/初始关节。
错误直接报出，**不会在运行中自动换成 PINN**。回退必须显式重新选择。
几何固定于初始臂/夹爪姿态，原有关节保持控制不变。

各后端的风预测使用当前自由关节的机体系角速度，避免 `mj_step` 后陀螺缓存滞后一帧。
日志位置、姿态、速度、角速度与关节从当前 qpos/qvel 同时刻读取。
未改 PX4 反馈/控制链路，也未新增 `mj_forward` 动力学计算。

参考的密度/Cd/几何是明确假设；旧 PINN 是历史模型对照，**不是独立真值**。
闭环差异反映模型选择的影响，不能据此宣称谁更接近真实飞机。
完整定义和限制见 [Task 1–2 报告](x4-950-reference-v1.md)。

## 先执行离线准备

启动任何 SITL 前：

```bash
source /home/raids/miniconda3/etc/profile.d/conda.sh
conda activate env_uav
cd /home/raids/Wind_field_simulation
python scripts/export_x4_950_parameters.py
python scripts/check_x4_950_reference.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg python -m pytest -q
python scripts/run_simulation.py --help
```

确认导出质量 8.0004 kg、29 个代理，零风增量/标签 Z 为零、测试通过。
这是离线检查，不连接 PX4。版本不匹配时先核对变更，勿重写散列绕过校验。

## 首轮：解析悬停，1 m/s +X 风

每个工况从相同 XML 初始状态、全新两个程序开始。终端一：

```bash
source /home/raids/miniconda3/etc/profile.d/conda.sh
conda activate env_uav
cd /home/raids/Wind_field_simulation
bash scripts/run_px4_sitl.sh
```

终端二：

```bash
source /home/raids/miniconda3/etc/profile.d/conda.sh
conda activate env_uav
cd /home/raids/Wind_field_simulation
python scripts/run_simulation.py \
    --wind-backend analytic \
    --reference-config outputs/pinn/x4_950_wind_v1/reference_config.json \
    --wind-x 1 --wind-y 0 \
    --run-log outputs/closed_loop/analytic_hover_w1_x_r1.jsonl
```

1. 确认控制台解析 reference 指纹，面板 `Wind backend: analytic`。此时仅有草稿，风仍未配置。
2. 等 `PX4: connected`，Unlock 并等 `armed`，再 Takeoff，等待 FLYING 和稳定悬停。
3. 无风稳定悬停至少 10 个仿真秒。确认 X=1、Y=0，Apply Wind；
   等接受/开关解锁后，点击 Wind OFF 变为 Wind ON。
4. 记下开风时面板 `Sim time`，保持恒定风悬停至少 **70 个仿真秒**。
   不改风速、不切轨迹、不点位置微调。默认丢弃开风后前 5 秒，分析随后 60 秒。
5. 关闭风扰，Land，落地后 Exit，确认两个程序回到提示符。
   EXIT 仍只管理本项目启动器创建的 SITL 子进程。
6. 汇总：

```bash
python scripts/summarize_wind_run.py \
    outputs/closed_loop/analytic_hover_w1_x_r1.jsonl \
    --settle-s 5 --duration-s 60
```

日志独占创建；复验用 r2/r3，不覆盖旧文件。异常退出或缺少完整窗口不能当成功结果。

## 旧 PINN 回退与同条件对照

新 PINN 候选可在第二终端这样选择，第一终端和其余操作相同：

```bash
python scripts/run_simulation.py --wind-backend pinn-x4 \
    --reference-config outputs/pinn/x4_950_wind_v1/reference_config.json \
    --pinn-checkpoint outputs/pinn/x4_950_pilot_v1/training_corrected/candidate.pt \
    --wind-x 1 --wind-y 0 \
    --run-log outputs/closed_loop/pinn_x4_manual_hover_w1_r1.jsonl
```

该新后端零风原始增量精确为0；更新风为0仍保留原XY滤波衰减，关闭风扰才重置。
完整气动函数的微小静止偏置和软物理约束限制见 [模型卡](models/x4-950-pinn-pilot-v1.md)。

落地 Exit 后重新启动终端一；终端二改为：

```bash
python scripts/run_simulation.py \
    --wind-backend pinn \
    --wind-x 1 --wind-y 0 \
    --run-log outputs/closed_loop/pinn_hover_w1_x_r1.jsonl
```

PINN 不传 `--reference-config`。其余点击顺序、风工况、时长、XML、轨迹参数、
面板/三维窗口、PX4 包/参数保持一致。汇总时换日志路径，比较完整指标。

无风基线：每个后端各新运行一次，文件名标记 `wind_off`，始终保持 Wind OFF，
FLYING 后至少 70 个仿真秒；汇总加 `--baseline`。OFF 时不预测，实际风力为零。
全零配置并 ON 可另检后端零风行为：解析增量严格为零，旧 PINN 沿用历史预测，
不能默认其零风输出为零。

## 扩展矩阵与复验

先完成无风和 1 m/s 悬停，再扩展，不必一次运行全矩阵：

| 维度 | 建议 |
| --- | --- |
| 后端 | analytic、pinn，独立日志 |
| 场景 | 悬停、Circle、Figure-8 |
| 合风速 | 0、1、2、3 m/s；零风方向去重 |
| 风向 | 0°、45°、90°、180°，表示气流在世界 XY 中吹向 |
| 重复 | 每工况至少 3 次，交换后端运行先后顺序 |

分量为 `[speed*cos(theta),speed*sin(theta),0]`。
3 m/s、45°可填两个分量均为 2.1213203435596424，不能填 `[3,3]`。
原气压噪声 seed=0，轨迹无新增随机操作。

轨迹对照：稳定悬停后选 Circle/Figure-8，先运行一完整周期再 Apply/开风。
两端显式传相同 `--trajectory-size 1 --trajectory-duration 20 --figure8-duration 30`。
开风后至少观察 70 个仿真秒，不在分析窗口切轨迹；汇总加
`--trajectory circle` 或 `--trajectory figure8`。60 秒窗分别覆盖 3/2 个周期。
手动点击不能保证逐步一致的开风相位，应报告实际时序和重复结果，不能称为严格重放实验。

## 日志与指标边界

日志由 metadata、50 Hz sample、closed 结束记录组成，默认不启用。
样本状态/目标/风力对应步前同一仿真时刻，步进后补写该次计算耗时。
metadata 包括后端、参数/代码散列、reference 指纹或原权重散列、质量、
滤波/限幅/轨迹设置及噪声 seed。它是对照观察记录，不是训练数据。

- `closed_cleanly`：异常、缺少结束标志、EMERGENCY 或记录的控制错误不可当正常完成。
- 位置 XY/3D RMSE、峰值和高度误差：使用各时刻实际目标；不从两条轨迹的加速度差生成标签。
- `force_z_max_abs_n`：须为精确零；原始预测 Z 另存诊断。
- `clipped_fraction`：不能靠限幅掩盖模型差异；同时保存原始/滤波加速度和实际力。
- `joint_max_abs_deviation`：joint1–6 为 rad，endleft/endright 为 m；偏离较大时冻结几何近似减弱。
- `outside_inspection_domain_fraction`：速度/倾角/角速度超出原检查域的比例，不是允许扩大风域或实物失效判定。
- 预测/迭代计算 P95：iteration 含桥接、控制、步进，不含随后日志写入、图形刷新及 pacing sleep；
  仅基于 50 Hz 样本，不是完整循环的硬实时保证。

`--settle-s` 仅裁剪时间，不自动判稳，也不验证裁剪前状态连续稳定；须按步骤保持恒定工况。
测量窗口本身会拒绝风开关/轨迹/飞行状态中断和不同配置风速混入。
保留原始日志；严重超域、关节偏离、控制饱和和异常工况分别报告。
