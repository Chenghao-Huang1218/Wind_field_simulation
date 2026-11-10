# Wind Field Simulation 项目约定

本项目用于测试和评估 PX4 SITL 在风场中的飞行表现，准备开源发布。

## 开始工作

- 先阅读 `docs/HANDOFF_2026-11-10.md` 和 `README.md`；涉及 PX4 启动或编译时再读 `docs/px4.md`。
- 当前项目根目录为 `/home/raids/Wind_field_simulation`。开发使用本项目内的代码、模型和参数。
- 本项目已从旧 PX4 工程独立；没有用户明确要求时，不修改 `/home/raids/PX4-Autopilot` 或旁边的 PINN 项目。
- 新会话先核对当前文件与交接状态，不重复迁移，也不自行假定后续功能需求。

## 开发规范

- 使用 Python 类型提示；关键函数、类包含 Google Style Docstrings。
- 变量命名清晰，遵循 PEP 8；深度学习默认使用 PyTorch。
- 新增实验随机操作设置 seed；数值计算优先使用 NumPy 向量化。
- 数学公式和算法实现参考经典论文或标准库；必要时注明来源。
- 复杂流程保留有用的关键步骤输出。修改前简要解释思路；发现逻辑或计算风险时直接指出。
- 已复制的第三方代码优先保持兼容，避免无关的格式重写。

## 验证与运行

本机 Python 环境为 `env_uav`。项目根目录下的离线验证命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg \
    /home/raids/miniconda3/envs/env_uav/bin/python -m pytest -q
```

- 离线测试通过与真实 SITL 闭环运行通过需要分别报告，不把进程替身测试当成飞行验证。
- 用户手动运行采用两个终端，均可显式激活 `env_uav`；完整命令见交接文档。
- EXIT 联动只管理本项目启动器创建的 SITL 子进程，保留会话隔离，不使用按进程名批量终止的方式。
- 修改 MuJoCo 模型时保留程序需要的 body、joint、actuator、sensor 名称，或同步更新绑定代码与验证。
- 公开发布前遵守 `THIRD_PARTY_NOTICES.md`；项目尚未确定所有材料的统一开源授权。
