# Cabinet 场景组件

来自 `/home/raids/cabinet model` 的 AM-Bench 柜子模型。
运行所需文件为 `cabinet.xml` 和 `textures/` 下的两张贴图。
源许可证、转换记录与原说明分别保存在 `LICENSE.source`、
`conversion_manifest.json`、`README.source.md`。

## 接入

`apps/configs/robot_quadrotor_950_with_arm.xml` 在已有模型之后 include 本组件，
以保留原有关节的 qpos/qvel 索引。纹理路径相对于本组件 XML。

接入时移除全局 compiler、option 和单自由度 keyframe，
并将 cabinet 根节点 pos 改为 `1.5 0 0`。由主场景管理编译和仿真选项，
保留主场景 0.005 s 时间步及积分器。
柜体固定于世界，正面朝 -X，主体约位于 x=2.1～2.72、y=-0.6～0.6、z=0～1.68 m。
如需调整位置，修改 cabinet.xml 中 `<body name="cabinet" pos="1.5 0 0">`。

保留全部 36 个碰撞几何和材质。滑动关节 `pull_door_slide` 保持被动，
范围约为 [-0.6, 0] m，默认 0（关门）；门把手参考点为 `door_handle`。
保留固定柜体与滑动门之间的接触排除，机器人仍可与柜体和门接触。
没有添加开柜门控制器。该组件继承源模型的质量等近似，详见原说明。

## 拉门演示调整

- 滑动关节 `frictionloss` 从 2 降为 0.2 N，`damping` 保持 10 N·s/m；接触摩擦不变。
- 从上往下数，第一、第二层的前门板及把手交换高度：
  `pulling_door`（包含门板、把手、关节和参考点）与固定半扇门
  `cabinet_geom_20` 上移 0.29 m；原顶层门板及把手
  `cabinet_geom_28`～`cabinet_geom_31` 下移 0.29 m。
- 活动门现在位于最上层，关闭时 `door_handle` 世界坐标约为
  `(1.98, 0.49, 1.52)` m，仍沿 -Y 滑开，行程约 0.6 m。
- 顶板、隔板、侧板、背板以及柜体总高度不变；门质量不变。
  这些几何调整不代表已验证实际抓取姿态下的旋翼避碰或飞行稳定性。

独立查看场景、演示/转换脚本、缓存和旧验证输出未复制。
