# X4-950 PINN Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for future approved implementation. This document itself does not authorize launching training or changing the running model.

**Goal:** 在水平风速 0–3 m/s 内，先从本项目 XML 派生独立气动参考，检查其直接用于仿真的行为与计算成本；确认需要代理网络后，再生成数据并从头训练适配 X4-950 当前整机配置的风扰 PINN。

**Architecture:** 冻结 XML/机械臂姿态 → 构建独立分部件阻力参考 → 同状态有风/无风配对标签 → PyTorch 机体系气动网络 → 差分获得风增量 → 现有水平施力路径。先进行离线验证，再做明确区分参考模型与网络的闭环对照。

**Tech Stack:** Python 3.12/env_uav、MuJoCo 3.6.0、NumPy、PyTorch；批量计算、NPZ/JSON、Matplotlib 科研图。

**Spec:** [X4-950 重训练设计草案](../specs/2026-11-10-x4-950-pinn-design.md)。当前计划为建议版本，用户已确定的数据来源为仿真、参数来源为 XML、风速范围为 0–3 m/s。

**最新会话结论：** 用户已接受“先建立并检查简化气动参考，再决定是否需要 PINN”。缺少经验证的气动模型时，可以生成基于明确假设的合成数据，但不能据此宣称真实飞机气动精度。后续工作移交新会话，优先执行 Task 1–2，不直接启动大规模生成、训练或运行模型切换。

## Global Constraints

- 只修改/开发当前项目；不修改原 `/home/raids/PX4-Autopilot` 或旁边 PINN 项目。
- 首版为当前默认 X4-950 + Airbot Play、初始机械臂/夹爪姿态、无新增载荷。
- 参考气动几何冻结于初始姿态；实际仿真保留原有关节保持控制，不新增锁臂或修改多体动力学。闭环记录关节偏离并声明这项近似。
- 水平合风速 0–3 m/s；输入 w_z=0，施力 F_z=0；保留 Apply-before-enable 和 EXIT 会话隔离。
- XML 参数事实与假设气动参数分开存档；不声称实物气动标定或 CFD 精度。
- 不用旧 PINN、EMA、限幅输出或不同闭环轨迹的加速度差作独立标签。
- 旧权重不覆盖；新权重、归一化器、数据和指标均使用版本目录和散列。
- NumPy/PyTorch 随机种子固定并记录；训练统计只由 train 拟合；最终测试集不参与调参。
- 新 Python 代码使用类型提示和 Google Style Docstrings；复杂步骤输出数量、范围、异常、散列和耗时。
- Task 1–2 交付独立离线参考和检查工具；当前应用、模型/XML、PX4、运行参数不变。

## Review Focus

1. 6.5 kg 机身与 8.0004 kg 整机质量混用；根点与质心速度/力混用。
2. XML 无法提供的 Cd/RPM 被伪装成已知事实，或 visual/collision/桨盘几何重复计数。
3. 有风/无风标签不在同一状态，或训练标签混入控制器、旧网络、滤波/限幅。
4. 新旧特征/权重不兼容、原 C4 对称性误用，或网络不保证零风零增量。
5. 行随机拆分造成泄漏、隐藏气动参数混合造成冲突，或闭环只比较限幅后的结果掩盖拟合误差。

## 现有参数与范围

已完成只读 [XML 审计](../../provenance/x4-950-xml-audit-2026-11-10.json)。当前机身质量 6.5 kg，整机 8.0004 kg，初始机械臂 joint1–6 均为 0 rad；步长 0.005 s。机身与整机惯量不同，后者详见审计 JSON。XML 推力上限与桥接归一化满量程约 39.242 N/电机分别存档，不混用；真实电机转速关系未知。

首版采样建议（这些是待验证的设计范围，不是已验证飞行包线）：

| 变量 | 采样范围/方式 |
| --- | --- |
| 水平风合速度 | 0–3 m/s，极坐标生成，所有方向 |
| 根点水平速度合速度 | 0–3 m/s，所有方向 |
| 根点垂直速度 | -1–1 m/s，用于覆盖起降/扰动；不等于允许垂直风 |
| roll/pitch | 各 -20°–20°；更大倾角不在首版声明范围 |
| yaw | -180°–180°，覆盖全方向 |
| 机体系角速度 | 合模长 ≤1.5 rad/s，含零角速度样本 |
| 机械臂/夹爪、载荷 | 固定初始配置，无新增载荷 |

相对空气速度必须覆盖风与机体速度相反的情况，不能把输入风 ≤3 m/s 错当成相对速度也 ≤3 m/s。

## Task 1：冻结 XML、气动几何与参考配置

**Files:** 新增 `src/wind_field_simulation/training/xml_parameters.py`、`configs/pinn/x4_950_reference_v1.json`、`scripts/export_x4_950_parameters.py`、`tests/test_x4_xml_parameters.py`。

**Interfaces:** `export_vehicle_parameters(xml_path: Path) -> dict[str, Any]`；输出整机/部件质量惯量、坐标变换、执行器/site、初始关节位置、几何代理清单和依赖散列。JSON 单位统一 SI，姿态 WXYZ，根 B 为 FLU、世界 W 为现有 ENU-like 坐标。

- [x] 测试先覆盖整机质量约 8.0004 kg、四旋翼 site 名称及位置、关节冻结、根点/质心分离；与 MuJoCo 编译结果比对。
- [x] 导出主场景递归 include 闭包和实际引用 mesh 的散列。现有审计仅是 XML 清单及初始聚合核对，不能替代完整 mesh/代理版本记录。
- [x] 为机身、机架/电机壳/起落架及固定机械臂建立去重的等效几何表；每项含 source、半轴、中心、方向、面积、质量归属与选择理由。
- [x] 排除 scene/floor/site/camera、重复 visual/collision 和实体桨盘误计。明确未建模桨叶尾流。
- [x] 冻结首版参考假设：ρ=1.225 kg/m³、逐轴 Cd=1.0，均标记 assumption；保留 Cd/几何变化为后续敏感性配置。

**验收：** 参数与编译 XML 一致；几何列表可逐项审查；没有未标记的经验参数或 Hummingbird/RPM 依赖。产物 `outputs/pinn/x4_950_wind_v1/vehicle_parameters.json`、`geometry_manifest.json`、`reference_config.json`。

## Task 2：独立参考气动力与配对标签

**Files:** 新增 `training/reference_aerodynamics.py`、`tests/test_reference_aerodynamics.py`。完整公式、坐标和首版近似见设计草案。

**Interfaces:** `reference_wrench_body(air_velocity_body: FloatArray, angular_velocity_body: FloatArray, reference: dict[str, Any]) -> tuple[FloatArray, FloatArray]`，支持批量；返回整机合力 N 与合力矩 N·m。`paired_wind_labels(...)` 返回两次完整参考力及水平风增量加速度。

- [x] 用手算单椭球/单轴例子测试量纲、符号和速度翻倍阻力四倍；测试零相对运动时无力。
- [x] 测试部件点速度 `v_O-w+ω×r`、坐标转换、净合力及力矩汇总；验证完整参考功率 `F·v_air+τ·ω≤0`。
- [x] 对同一 s 计算 `F_ref(s,w)` 和 `F_ref(s,0)`；标签为 `P_xy ΔF/m_total`，不做速度数值差分。
- [x] 测试 w=0 精确零增量、合风速上限、非零垂直风拒绝、原姿态/控制状态保持一致。
- [x] 标称解析参考模型可直接作为 baseline；PINN 是否带来速度/接口优势需测量，不预设优于解析计算。

**验收：** 独立于旧 PINN/权重；物理性质与坐标测试通过；风增量的耗散性不被错误要求为 `ΔF·v_air≤0`。首版风增量可向车辆输入能量，不能对它套用完整无源阻力约束。

**进入训练前的决策点：** 先记录解析参考的直接计算耗时、行为与适用边界。如果它已经满足仿真需求，就保留解析参考方案；若训练代理网络，明确其实际用途和可验证收益。不得预设网络更快或更准确。Task 3–6 为选择继续 PINN 路线后的后续步骤。

## Task 3：生成 0–3 m/s 训练/验证/测试集

**Files:** 新增 `training/sampling.py`、`training/dataset_v1.py`、`scripts/generate_x4_wind_dataset.py`、`tests/test_wind_dataset_v1.py`。

**Interfaces:** `generate_dataset(config_path: Path, output_dir: Path) -> Path` 返回 manifest 路径。每行保存 sample/group ID、种子、v_O^W、WXYZ 姿态、ω^B、w^W、臂姿态版本、full/no-wind 参考力、水平增量标签和参考配置散列。完整原始三轴力与应用的水平标签分开。

- [ ] 先做独立 3,000 样本 pilot（seed=7），检查范围、方向/速度分布、NaN、标签峰值、生成耗时与内存；pilot 不用于最终测试。
- [ ] 明确数据来源为参考模型的离线状态扫描；不将它描述为真实飞行或已完成的 SITL 轨迹采集。原始完整气动力与有风/无风配对均保留，供完整量和增量损失使用。
- [ ] 正式建议 120,000 对：train 80,000 / val 20,000 / test 20,000；分别使用 seed 42/43/44。分批 NumPy 向量化生成，保存分片 NPZ 和 manifest。
- [ ] train 风速锚点 `{0,0.5,1,1.5,2,2.5,3}`，风向每 30°；val 用 `{0.25,0.75,1.25,1.75,2.25,2.75}`、风向偏移 15°；test 用独立速度/状态及 7.5°、22.5°偏移风向，并包含独立状态下的 0/3 m/s 边界。
- [ ] 其他状态按上表分层抽样；建议 10% 零角速度，10% 悬停邻域，显式含零风样本与相反相对来流。
- [ ] 按工况/采样组隔离，不拆分同一配对；禁止跨集合重复状态/样本，并检查坐标变换后的机体系核心输入及配对，避免只靠世界坐标不同掩盖重复。以后若补闭环轨迹，按整段 episode 划分。
- [ ] 标称集合只用冻结的一组气动参数；Cd 0.7/1.0/1.3 倍、代理尺寸±10%仅生成分离的敏感性评估集，所有风仍 ≤3 m/s。这些变化不是已知真实参数置信区间。

**验收：** manifest 可复现；归一化仅从 train 拟合；无泄漏/重复/风速越界；生成分布、力幅值、零风对照报告。不得为让现有 6 m/s² 限幅好看而截断原始训练标签。

## Task 4：从头训练新的物理约束网络

**Files:** 新增 `training/model_v1.py`、`training/train_v1.py`、`scripts/train_x4_wind_pinn.py`、`tests/test_wind_model_v1.py`。

**Interfaces:** `X4WindAeroV1.forward(air_velocity_body: Tensor, angular_velocity_body: Tensor) -> Tensor`，输入两项均 `[N,3]`，输出完整机体系参考气动加速度 `[N,3]`。保留 6D 机体系归一化统计，g=非负系数二次阻力先验+残差 MLP `6→64→64→3`。

- [ ] 先测试输出形状、训练统计复用、零风差分恒等、坐标协变和拒绝旧 17D 权重。先验系数由标称参考的单位轴向样本拟合，存档而不称实测。
- [ ] 数据损失同时拟合完整参考气动加速度和当前应用的水平风增量，误差按 train 的每轴尺度归一化（尺度下限 0.05 m/s²）。
- [ ] 在 ω=0 子集对完整气动力加速度加入正功惩罚，初始权重 0.01；不对风增量套耗散约束。不复用旧 C4 损失/XY 统计对称化。
- [ ] 初始配置：PyTorch float32、AdamW lr=1e-3、weight_decay=1e-5、batch=1024、最多 200 epoch、val patience=20；ReduceLROnPlateau factor=0.5/patience=5，梯度裁剪 1.0。这些是可由 val 调整的起点。
- [ ] seed=42 主实验；43/44 复验稳定性。只按 val 选超参数/epoch，最终 test 保持锁定。训练/采样种子及确定性设置、设备版本均保存。
- [ ] checkpoint 包含 kind=`x4_950_wind_increment_xy_v1`、网络/特征版本、单位与坐标、XML/mesh/参考/数据散列、质量、臂姿态、风域、先验、归一化、种子与验证指标；旧 root 权重不覆盖。

**建议数值验收门槛（执行前冻结）：** 原始、未 EMA/限幅的水平风增量每轴 RMSE≤0.05 m/s²，向量误差 P95≤0.10 m/s²；报告均值/最大值及风速/方向/姿态分组误差。相对误差仅在参考幅值足够时报告。零风增量和施力 Fz 必须为精确 0。门槛不是已达到的结果；未通过时保留旧运行版本并记录失败。

同时建议非零风集合的整体归一化误差 `sqrt(mean(||e_xy||²))/sqrt(mean(||y_xy||²))≤5%`，另报恒零预测 baseline，防止标签幅值很小时仅靠宽松绝对门槛通过。分母接近数值零时不报告该比值，改列绝对误差；全部门槛仅可在 pilot 后、正式训练/测试前冻结，不根据 test 结果放宽。

## Task 5：离线验证与候选闭环对照

**Files:** 新增 `training/evaluate_v1.py`、`scripts/evaluate_x4_wind_pinn.py`、`tests/test_x4_wind_integration.py`。后续闭环入口放在本项目 scripts 中，不修改飞控源码。

- [ ] 比较：独立解析 teacher、简单拟合二次阻力 baseline、新 PINN；旧 PINN仅作历史参照，不作真值。
- [ ] 分别报告 full/increment 误差、零风行为、坐标性质、CPU/GPU 数值一致性、预测耗时及参考生成耗时；不凭网络形式宣称更快。
- [ ] Cd/几何敏感性单独报告。把它与实物准确性分开；变参偏差不是同一标称代理网络的拟合误差。
- [ ] 原始网络验收后才进行 teacher/PINN 闭环 A/B：同一 XML、初始状态、PX4 参数、轨迹、风配置、根 body 施力点、XY 投影、EMA 和限幅。teacher 在 scratch 状态评价净风增量，再通过现有单点 force-only 入口施加；不能一边启用分布式全气动力/力矩、一边只用单点水平力。
- [ ] 两端均使用同一冻结的参考臂几何，并记录实际臂/夹爪关节偏离；若需要覆盖臂姿态变化，另建包含关节输入的版本，不能只在 teacher 中启用动态几何。
- [ ] 闭环工况建议：悬停、Circle、Figure-8；风速 `{0,1,2,3}`，方向 `{0°,45°,90°,180°}`，去重零风；先稳定悬停再开风。运行顺序与随机扰动固定种子，重复至少 3 次。
- [ ] 记录位置 RMSE/峰值、高度偏差、姿态/控制饱和、原始与实际风力、限幅占比、臂关节偏离和超域样本。只在 teacher 可完成的工况评价替代误差；teacher 也失败的工况标记不可运行，不据此单独判定 PINN。

**建议闭环替代门槛：** 相对 teacher 的位置 RMSE 增量不超过 `max(10%, 0.02 m)` 所对应的绝对差（即 `RMSE_PINN-RMSE_ref ≤ max(0.1*RMSE_ref,0.02m)`），无新增异常退出；同时报告完整结果，不能用限幅掩盖离线误差。该门槛是计划建议，非飞行保证。

## Task 6：接入、文档与版本发布准备

**Files:** 修改 `simulation.py` 的模型加载适配及 README/handoff；新增 `assets/checkpoints/x4_950_wind_v1/` 候选目录，保留现有 checkpoints。

- [ ] 保持 `predict(velocity_world, quaternion_wxyz, angular_velocity_body, wind_world)` 接口；新 wrapper 批量评价 g(v-w) 与 g(v)，返回世界系三轴风增量供原始诊断；现有施力函数继续执行 XY 投影/滤波/质量换算。
- [ ] 新模型不再使用 Hummingbird 参数、固定悬停 RPM 或旧 normalizer。当前 XML 无独立旋翼气动参考，因此首版不能声称学到了 RPM/推力诱导气流作用。
- [ ] 从 metadata 验证质量/坐标/几何/臂姿态及模型版本；新模型风输入合速度超过 3 m/s 时明确拒绝，不静默截断；不支持的状态范围明确列在模型卡中。
- [ ] 实现显式 checkpoint 选择和旧模型回退。离线/闭环验收前旧模型仍为默认；验收通过后再将候选设为默认，并保存回退命令。
- [ ] 保留空白启动、成功 Apply 才解锁、无效输入保留设置、运行时改风、全零风、Z=0、位置微调和 EXIT 进程隔离的回归测试。
- [ ] 模型卡写明仿真参考来源、经验参数、固定臂姿态、0–3 m/s、未建模效应、测试误差/时延及无实测验证。发布前单独核对新参考代码/数据/权重授权。

**最终产物：** 参数/参考配置与散列、120k 配对数据及 manifest、新 PyTorch 权重/normalizer、离线和闭环报告、模型卡、可选的新旧加载入口。除已完成的 XML 审计和本计划外，以上均尚未实施。

## 执行顺序与停止条件

先 Task 1–2 冻结参数与 teacher，并检查直接计算方案；确认需要 PINN 后，再 pilot 和正式数据、训练/离线测试，最后闭环和接入。无 wall-clock 承诺；pilot 后记录每阶段实际成本。

若几何去重不清楚、teacher 不满足自身性质、无风增量不为零、XML/mesh版本不一致或标签依赖旧网络，停止进入训练；先修正参考定义。若新网络未过离线门槛，不切默认、不靠增大限幅掩盖误差。若要覆盖机械臂动作或旋翼流场，需要扩展参考与输入，并作为新版本设计。

## Progress

- [x] 读取当前 XML/运行映射并生成参数审计；确认用户风速范围为 0–3 m/s。
- [x] 完成研究设计与任务计划草案。
- [x] 完成独立可执行性审阅；未发现需阻塞实施的关键缺口，补充固定参考几何与实际关节保持的区别。
- [x] Task 1–2：冻结参数/去重代理、实现独立参考与配对标签，通过离线检查；见 [结果记录](../../x4-950-reference-v1.md)。
- [x] Task 3–4 的用户授权 pilot：3000 对、独立 2000/500/500 分组、seed7，
  seed42 CPU 200 epoch。离线误差/零风/耗时已测；见
  [pilot 记录](../../X4_PINN_PILOT_2026-11-10.md)。正式120k/多seed未执行。
- [x] Task5–6的pilot运行部分：可选pinn-x4接入、离线220项通过、悬停OFF/+X1/2/3
  两后端各3重启，24/24真实SITL完整完成；默认旧PINN和解析保留，未修改PX4。
  [结果与范围](../../X4_PINN_INTEGRATION_2026-11-10.md)。正式训练/其它风向轨迹/
  GPU一致性/敏感性及默认切换未包含在本次任务内。
- [x] 后续仅解析路线：Task 5–6 的可选后端/日志/离线部分及首轮真实悬停 SITL（12/12）已完成；默认仍旧 PINN，未训练或接入新网络。见 [后端实施计划](2026-11-10-analytic-wind-backend.md) 和 [首轮结果](../../SITL_FIRST_ROUND_2026-11-10.md)。
