# Prompt-Conditioned Constraint for End-to-End RL Navigation

## 1. Motivation

为什么值得做？

- 现有的端到端导航仅优化单一目标，难以兼顾复杂的动态高层语义约束（如靠右行驶、避开厨房、礼让行人）
- 传统方法调整行为需重新设计 Reward 或重新训练 Policy，成本高且泛化性差
- 尽管 VLM 和 LLM 广泛用于高层 Task Planning，但极少直接作用于底层的 RL 控制
- 利用 Prompt 动态约束导航策略，在无需重训 RL Policy 的前提下，根据用户多模态指令实时调整机器人行为

---

## 2. Core Idea

你打算怎么做？

- 冻结已有的端到端导航策略，仅训练一个由 Prompt 引导的 Semantic Controller，将原始动作空间或特征投影到符合语义约束的流形中
- Prompt 不直接生成动作，而是定义当前允许的行为流形，控制器负责将 RL 输出映射至该空间内

---

## 3. Research Gap

别人没有解决什么？

现有工作多关注 Prompt 控制任务规划、VLA 模型直接生成动作或基于规则的安全约束，但存在以下不足：

- 缺少无需重训策略即可动态改变底层导航行为的方法。
- 通常依赖硬编码规则或复杂的奖励工程，缺乏统一的可学习框架。
- 尚未充分研究 Policy Latent Space (策略隐空间) 与语言语义空间之间的映射关系，未将 Prompt 建模为动态的连续约束流形

---

## 4. Novelty

1. 提出 Prompt-Conditioned Semantic Manifold (提示词条件语义流形)，将自然语言转化为动态策略约束空间，而非直接生成动作
2. 引入可学习的 Projection Controller，实现对 Frozen Policy 的零重训即插即用控制
3. 支持硬约束（如禁止进入、停止）与软约束（如尽量靠右、保持速度）
4. （可能）支持文本、音频、场景语义及目标特征联合生成行为约束

针对投影算子: Prompt 生成投影矩阵 $P$，通过 $z' = Pz$ 实现几何投影，参数量小、可解释性强

硬约束：将动作严格投影至流形边界内
软约束：在目标函数中加入语义惩罚，最小化偏离距离

---

## 5. Related Work

- 端到端 RL 导航、可提示机器人、VLA、安全强化学习、控制屏障函数、隐空间操控
- 传统方法沿用 `Prompt -> Planner -> Action` 链条；本工作采用 `Prompt -> Semantic Constraint -> Frozen Policy -> Action`，将 Prompt 视为行为空间的约束，而非直接的控制指令

---

## 6. Feasibility

| | |
| --------- | ----------------------------------- |
| Data | 模拟器在线生成 + LLM/人工 Prompt |
| Simulator | Isaac Sim |
| Code | 对现有 RL 框架增加 Policy Projection |
| GPU | RTX 4070 |
| Difficulty | 5 |

---

## 7. Expected Contribution

- 理论：提出 Prompt 条件下的语义流形策略约束框架
- 算法：设计出无需重训策略的通用可学习投影控制器
- 工程：实现零重训成本下的机器人导航行为实时动态切换
- 实验：验证了不同 Prompt 约束下导航行为的一致性与可控性

---

## 8. Risks

1. [!] 离散的语言语义难以平滑映射至连续的策略流形
2. [!] 策略投影设计及收敛困难，设计不当可能引起控制指令错乱
3. 硬约束过于严格时可能导致策略无解
4. 控制器表征能力不足，导致无法完美保持 RL 原有的性能

---

## 9. Next Step

1. 学习 Prompt Robotics 与 Policy Constraint
2. 复现基础的端到端视觉导航 Baseline
3. 搭建 Prompt Encoder 并实现不同的 Projection Controller 方案进行横向对比

---

## 10. Reviewer Questions

- Q1：既然完全不重训 RL，策略如何具备理解并适应复杂新 Prompt 的泛化能力？
- Q2：相比传统的 Reward Shaping 或基于规则的控制，该方法的优势和效率体现在哪？
- Q3：[!] 为什么不直接用大容量的 VLA 或 LLM 直接端到端输出底层 Action？
- Q4：Prompt 约束为什么应该作用于底层的 Policy 特征，而不是高层的 Planner？
- Q5：在几何或优化视角下，为什么 Projection 会比 Residual 更加合理？
- Q6：[!] 抽象的 Semantic Manifold 究竟是如何被量化并由网络学习到的？
- Q7：如何通过实验证明 Controller 学到的是通用“语义约束”，而非针对特定场景的“新导航策略”？
- Q8：[!] 投影操作如何从理论或实验上保证不破坏原策略的最优性/安全性？
