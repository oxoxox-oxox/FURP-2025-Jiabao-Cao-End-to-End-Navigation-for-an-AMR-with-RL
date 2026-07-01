## 一、端到端导航领域整体局限性（大范围背景）

| 序号 | 问题 | 简述 |
|---|---|---|
| 1 | 因果混淆 | 模型学到虚假相关而非真因果 |
| 2 | 开环/闭环评测不一致 | 训练表现好≠真实部署表现好 |
| 3 | 安全性与可解释性 | 黑箱决策难满足安全认证 |
| 4 | 推理速度/实时性矛盾 | 大模型/复杂优化跟不上控制频率 |
| 5 | Sim-to-real具身鸿沟 | 仿真训练效果难迁移真实硬件 |
| 6 | 长时序记忆缺失 | 局部最优/卡死 |
| 7 | 多模态指令泛化 | VLA类方法指令理解能力有限 |

---

## 二、NeuPAN官方承认的四条局限性

来源：NeuPAN GitHub仓库 README "Current Limitations" 章节。

| 序号 | 局限性原文 | 中文 |
|---|---|---|
| 1 | CPU-bound optimization: The NRMP layer uses cvxpy which does not support GPU acceleration | NRMP层用cvxpy，不支持GPU加速，推理速度受CPU限制 |
| 2 | Supported kinematics: Currently limited to differential drive, Ackermann, and omnidirectional robots | 仅支持差速/阿克曼/全向轮三种运动学 |
| 3 | Convex robot geometry: The DUNE model assumes convex robot shapes | DUNE假设机器人是凸形，非凸需凸包近似 |
| 4 | Parameter tuning: Performance in specific environments may require tuning the `adjust` parameters | 特定环境下需手动调参 |

---

# 方案一、NeuPAN腿足扩展

不修改NeuPAN的DUNE/NRMP核心数学公式，只在"用哪个训练好的形状模型"这一层做文章：训练两个独立的DUNE模型（对应两种典型步态形状），外层加一个简单的切换逻辑。

NeuPAN 官方明确表示腿足等其他运动学需要自己改约束，且经检索，**目前没有公开工作把 NeuPAN 的 DUNE+NRMP 框架应用到腿足机器人的底盘运动规划上**——已有的腿足避障工作都走的是经典 MPC 或纯 RL 路线，没有人用 NeuPAN 这种"点云直接映射隐式距离特征+可微凸优化"的范式做腿足导航。

将 NeuPAN 的端到端模型驱动避障框架扩展到腿足机器人底盘运动学约束，对比纯 RL / 纯 MPC 方法在可解释性、训练效率、sim-to-real 泛化上的优势：

1. **可解释性**：几何特征显式建模，决策逻辑可追溯；
2. **训练效率**：避免大规模RL试错与昂贵采样；
3. **Sim-to-real泛化**：隐式距离特征的迁移性优于端到端策略黑箱。

| 步骤 | 内容 | 预计时间 |
|---|---|---|
| 1 | 量取机械狗两种典型形状的尺寸参数（双脚支撑/单脚迈步） | 1天 |
| 2 | 用NeuPAN官方`example/dune_train`脚本，分别训练形状A、B两个DUNE模型 | 2-3天 |
| 3 | 编写"何时用A、何时用B"的切换逻辑（先用固定时间节奏的朴素版本） | 3-5天 |
| 4 | 训练baseline（取A、B并集的保守包络形状）模型 | 1天 |
| 5 | 在ir-sim设计窄道测试场景，跑对照实验（动态切换 vs 保守包络） | 1周 |
| 6 | 整理数据、调参、补实验、写报告 | 1周 |



# 老师指导方向梳理与可行性评估

## 方向一：灾难性遗忘（Catastrophic Forgetting）：学会技能A，学B不遗忘A

   之前老师尝试过：弹性权重更新（EWC），给每个权重计算一个"重要性分数"（用Fisher信息矩阵估计），学新任务时对重要权重加约束，不让它变化太大。
   claude评价：**可行性：中等偏高。** EWC实现不复杂（在loss里加一项正则化），和你现有代码兼容，但需要额外计算Fisher信息矩阵（在标准环境上跑一次，记录梯度）。

---

### 方向二：路径中带值的非MDP

**解释A：历史依赖（Non-Markovian）**

标准MDP假设当前状态包含所有信息（Markov性质），但现实中"路径历史"很重要——比如机器人知道自己刚从哪里来，就不会再回头走死路。你的RCPG（GRU）就是在尝试引入历史信息，但这会破坏Markov性质，严格来说不再是MDP了。这个方向的核心问题就是"如何在非Markov环境下还能有效训练"。

**解释B：路径奖励/选项框架（Options Framework）**

MDP里每一步都有即时奖励，但如果"完成一段路径"才给奖励（比如"成功绕过U形墙"算一个选项），这就是HRL（层次化强化学习），把长路径分解成子目标。

claude评价：**可行性：低。** 这个方向理论性太强，实现周期长

---

### 方向三：POMDP（部分可观察马尔可夫决策过程）

**"只有部分环境，环境缺失"就是POMDP。** 标准MDP假设agent能完全观察当前状态，但现实中激光雷达只能看到前方180°，背后有什么不知道——这就是"部分可观察"。POMDP是处理这种信息缺失的框架。

claude评价：**可行性：中等。** POMDP的经典解法是维护一个"belief state"（对当前状态的概率分布估计），或者用RNN隐状态近似belief。你的RCPG已经是POMDP的一种近似实现——GRU的隐状态可以理解为对历史的压缩编码，相当于一个隐式的belief state。

