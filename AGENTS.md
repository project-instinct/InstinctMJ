# Agent 指南 - InstinctLab 到 mjlab 迁移（InstinctMJ）

## 作用范围

- 本指南适用于当前仓库下的全部内容。
- 除非用户明确要求跨项目修改，否则改动应限制在本项目内。
- `../InstinctLab/**` 与 `../mjlab/**` 默认仅用于参考。
- 未经用户明确要求，不得修改 `../instinct_rl/**`。

## 核心目标

- 在 `InstinctMJ` 内以 mjlab 原生方式复现 InstinctLab 行为。
- 文件/模块命名在可行范围内尽量贴近原 InstinctLab 结构。
- 保持本项目可作为 `instinct_rl` 的即插即用组件。
- 迁移时优先参考 `../mjlab/**` 中相似实现，采用其原生接口与组织方式。

## 迁移规则

- 优先复用 `InstinctLab` 的现有代码与类层级；若原实现使用继承，迁移后也尽量保留相同继承关系。
- 不得无依据新增“补充函数/辅助函数/包装函数”；原版能直接对应的函数，迁移后应保持同等函数数量与职责划分。
- 函数迁移按“原版一一对照”执行：原版几个核心函数实现，迁移后默认也保留几个核心函数实现，不额外拆分或合并。
- 不得以“代码更简洁/更优雅”为由压缩原版实现步骤；应按原版逻辑展开，避免隐藏中间过程。
- 在不破坏兼容性的前提下可复用原逻辑代码，重点改写为 mjlab 可接受的表达方式。
- 禁止新增兼容层/桥接层（compat layer、adapter shim、过渡包装）。
- 兜底逻辑必须与源任务一致：源任务有则保留，源任务无则禁止新增。
- reward 必须改为 mjlab reward-term 表达；其他 manager 若存在表达差异，也按 mjlab manager/config 模式迁移。
- 地形及其他核心功能需保持行为一致（reset、command、观测/动作、终止、课程学习）。
- 迁移后必须清理 旧框架 接口残留（import、类型、字段、注释、路径）。
- 对 InstinctLab 特有命名保持稳定（例如 `hack_generator`），仅在 mjlab 必要字段映射处改名。
- 清理无效旧路径、过期注释和迁移遗留内容。

## 旧框架到 mjlab 对齐要点

- 使用 mjlab 对应导入和命名（如 `ManagerBasedRlEnvCfg`）。
- 将 `@configclass` 配置迁移为字典式 manager 配置对象。
- 以下 manager 均采用字典表达：`rewards`、`observations`、`actions`、`commands`、`terminations`、`events`、`curriculum`。
- 场景使用 `SceneCfg` + MuJoCo 资产配置；不保留 USD/`prim_path` 方案。
- 传感器按 mjlab 方式配置并挂接到机器人实体。
- 统一使用 `entity_name` 字段，不使用旧字段名。

## 兼容性约束

- 将 `InstinctMJ` 视为 `instinct_rl` 的组件层。
- 避免破坏 `instinct_rl` 使用到的训练/回放入口与配置接口。
- 若修复必须改动 `instinct_rl`，先停止并询问用户。

## 代码质量

- 遵循当前项目风格：2 空格缩进、显式类型标注、配置/数据类为中心的结构。
- 保持代码清晰、模块化，避免无关重构。
- 迁移过程中清理无用 import 与死代码。
- 交付前尽量运行可用的 lint/format/tests。
