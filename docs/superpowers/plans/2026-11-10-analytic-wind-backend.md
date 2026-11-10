# Optional Analytical Wind Backend Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for the user-authorized inline implementation; keep a final independent review.

**Goal:** 接入已通过 Task 1–2 的解析参考，显式可选，默认旧 PINN；先离线验证，再提供 SITL 对照步骤。

**Architecture:** 已有 `predict(velocity_world, quaternion_wxyz, angular_velocity_body, wind_world)` 作为唯一后端接口。解析 wrapper 校验冻结 reference/vehicle/geometry 与当前 XML，返回同状态风增量；原 XY 投影、EMA、逐轴限幅、质量换算和根 body 单点 force-only 施力链路共用。后端仅启动时选择，回退用显式 `pinn`，错误不静默换模型。

**Spec:** 用户本轮明确请求及 [既有参考设计](../specs/2026-11-10-x4-950-pinn-design.md) 的接入边界；本计划替代其中“必须先训练才能接入”的顺序，仅接入解析候选，不训练。

**Tech Stack:** 当前 Python/env_uav、MuJoCo、NumPy 和原 PINN/PyTorch。

## Global Constraints

- 水平合风速 0–3 m/s，两个后端的 CLI、面板、SET_WIND 共用边界；w_z=0、实际 F_z=0。
- 默认 `--wind-backend pinn`；解析为 `--wind-backend analytic --reference-config PATH`。不覆盖旧权重。
- 无风关闭重置原滤波；风参数更新保持 XY 滤波与飞行目标、开关状态；保持 Apply-before-enable 和 EXIT 会话隔离。
- 参考固定 qpos0 机械臂/夹爪；实际原有关节保持控制不变。可选 JSONL 记录偏离、输入域标志、原始/滤波/实际施力及计算耗时，采样 50 Hz，不作训练数据。
- 只改当前项目 Python/文档；不改 XML、PX4 源码/参数、不启动 SITL、不训练。

## Review Focus

1. 错配 XML/质量/关节或改坏的参考产物必须在创建连接前失败。
2. 解析模型不能加载旧权重、使用活动几何或把全气动力/力矩额外施加。
3. 两后端统一风域；非法 Apply 保留已接受配置，CLI 草稿不能启用。
4. 原始 Z 可用于诊断，但实际施力必须水平、无旧 torque；不同后端不更改滤波参数。
5. 对照日志状态与施力同一时刻；不得覆盖旧日志、掩盖异常退出、把离线替身当飞行验证。

## Task A: Validated reference loader and predictor

- [x] RED：新增 `tests/test_wind_backends.py`，覆盖产物/当前 XML/质量/代理一致性拒绝，predict 与独立 paired 结果、零风和错域输入。
- [x] GREEN：共享 `load_reference_bundle(...)` 放入 `training/xml_parameters.py`；检查脚本复用，新增 `wind_backends.py` 解析适配器。
- [x] VERIFY：针对上述测试和现有参考测试运行 pytest，预期全绿。

## Task B: Runtime selection, shared domain and optional comparison logs

- [x] RED：实际 MuJoCo app + 无网络 bridge，覆盖两后端选择、无需旧权重、原施力链路/滤波/限幅/更新/开关；新增域门控测试，旧用例的超域有效值改为域内值，保留原测试目的。
- [x] GREEN：CLI/SimulationConfig/backend 工厂，面板显示后端；新增可选 `--run-log PATH` 的 JSONL metadata/50Hz sample/close summary，记录实际全部8个臂/夹爪关节、时序、误差、原始/实际力与限幅/域标志。默认无日志。
- [x] VERIFY：聚焦与完整离线 pytest；CLI help 与独立解析 app smoke，不开 socket/GUI/PX4。

## Task C: Documentation and review

- [x] README/交接同步当前行为；新增双终端、显式回退、无风/解析/旧PINN对照矩阵与日志分析命令；声明旧PINN是历史对照而非独立真值。
- [x] 独立只读审阅，修正阻塞缺陷，再做最终全量验证；不启动真实 SITL。

## Execution ledger

- 本轮已有明确实施授权，继续既有设计的解析接入子集；不另行要求设计审批。
- 目录尚无 Git 元数据，继续原地执行，不初始化仓库或创建工作树；使用此记录替代依赖 Git 的执行脚本。
- A→B：自包含 reference + 相同四输入 predictor；同一原施力链路，返回单位 m/s²，质量必须一致。
- B→C：日志明确后端和 reference/权重/运行配置来源，人工交互时序差异限制写入对照步骤。
- A complete：校验加载器/解析 predictor 与原参考测试通过；运行时不加载旧权重。
- B complete：两后端共享域、施力、滤波/限幅，日志独占/异常关闭/同步采样测试通过。
- C complete：195 passed in 24.12s；CLI help、无连接解析初始化/面板渲染/清理通过。未启动 SITL。
- Final review：1 项 P2 时间混用已修复；两个回归用例均 RED→GREEN，日志读取当前 qpos/qvel，风预测取当前根角速度，不改 PX4 反馈链路。
- Final minor (deferred)：settle-s 仅时间裁剪，不验证裁剪前连续稳定；对照说明要求保持完整恒定工况，不能当成自动判稳。
- 审阅不作判断的实物气动精度、真实 SITL 性能、恶意全量重写指纹均不扩大为本次保证；校验用于来源/误配检查，不是签名认证。
- 后续用户明确选择真实 SITL：首轮悬停 OFF/1 m/s +X、两后端各3次，正式12/12完成，详见 [结果](../../SITL_FIRST_ROUND_2026-11-10.md)。
- 实际降落暴露 HIL int16 封包越界，仅 Python 桥接编码前饱和，真实编解码两例 RED→GREEN，全量197通过；未改PX4/原参数。独立审阅保留接收端既有缩放限制。
- 运行选择记录：采用生产循环的无GUI驱动及官方实例6隔离，不能扩称GUI鼠标验证；3次失败探测全部排除；预定60秒主结果保留，额外50秒稳定阶段仅补充说明过渡差异；训练仍不执行。
