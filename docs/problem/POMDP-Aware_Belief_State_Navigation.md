# Research Topic: POMDP-Aware Belief State Navigation for LiDAR-Based AMR

## 1. Motivation
为什么值得做？

- 几乎所有现有的端到端LiDAR RL导航工作都将问题建模为标准MDP，假设当前观测（LiDAR扫描+目标方向）已经包含所有决策所需的信息。但现实中**这个假设是错误的**：
  - 2D LiDAR只能观测前方180°，机器人背后的障碍物信息完全缺失
  - 在U形陷阱中，机器人无法从单帧LiDAR中判断"自己是否已经进入了凹形结构"——只有结合历史观测才能推断出这个全局信息
  - 在对称走廊中，左右两侧LiDAR返回完全相同，单帧观测无法区分"应该向左还是向右"
- 我们的预实验已经直接证明了这一点：
  - CNNTD3（无记忆，单帧观测）在U-trap中SR=0%——它无法推断自己进入了陷阱
  - RCPG（GRU，10步历史）在窄门中SR=75%（比CNNTD3更好），但在双U中SR=0%（比CNNTD3更差）
  - 这说明简单加记忆（GRU）不能正确解决POMDP问题，需要更有针对性的信息融合机制
- 虽然很多论文在"问题建模"部分写了POMDP公式，但实际算法设计中并没有显式利用部分可观察性——GRU/LSTM只是作为特征提取器使用，没有建模belief state

---

## 2. Core Idea
你打算怎么做？

**核心思路**：显式构建一个轻量级的Belief State表示，补充单帧LiDAR缺失的全局信息，作为额外输入送入策略网络。

**Belief State包含三个组件**：

**组件1：局部占用记忆图（Local Occupancy Memory）**
- 维护一个以机器人为中心的小型2D栅格地图（如20×20，分辨率0.5m，覆盖10m×10m）
- 每一步将当前LiDAR扫描投射到该栅格地图上，标记"已知占用"、"已知空闲"、"未知"
- 随机器人移动，栅格地图随之平移和旋转（egocentric视角）
- **关键作用**：机器人转身后仍然"记得"背后有墙，弥补LiDAR 180°视角的盲区

**组件2：历史轨迹编码（Trajectory Encoding）**
- 记录最近N步（如50步）的机器人位置序列
- 用一个轻量级的1D CNN或简单的统计特征（质心、散布度、回环检测）编码
- **关键作用**：检测"是否在绕圈/是否已经进入U形结构并试图原路返回"

**组件3：时间步计数器（Temporal Context）**
- 当前episode已经过了多少步
- **关键作用**：给策略提供"是否快要超时"的紧迫感，在接近超时时触发更激进的探索

**策略网络输入**：
$$s_{belief} = [\underbrace{l_1...l_{180}}_{\text{LiDAR}}, \underbrace{d, \cos\theta, \sin\theta}_{\text{goal}}, \underbrace{M_{20\times20}}_{\text{占用记忆图}}, \underbrace{h_{traj}}_{\text{轨迹编码}}, \underbrace{t/T}_{\text{时间比例}}]$$

**训练方式**：仍然用TD3，只是状态空间扩大了，网络结构需要增加一个小型CNN分支处理占用记忆图。

---

## 3. Research Gap
别人没有解决什么？

- **POMDP公式化 ≠ POMDP解法**：大量LiDAR导航论文在Section 2写了POMDP公式，但实际算法只是标准MDP的策略梯度/Actor-Critic，没有显式建模belief state。GRU/LSTM作为隐式belief近似，但其有效性**从未在结构化困难场景中被系统评估**
- **轻量级Belief State在LiDAR导航中的缺失**：现有显式belief state方法（如QMDP-Net、FORBES）针对视觉导航或自动驾驶，使用复杂的生成模型（Normalizing Flow、Transformer），不适合资源受限的AMR平台
- **DreamerNav（2025）**将POMDP+世界模型用于导航，但依赖深度相机和全局占用图，不是纯LiDAR端到端方案

---

## 4. Novelty
真正的创新点是什么？

1. **轻量级Belief State设计**：提出基于局部占用记忆图+轨迹编码的显式belief表示，不需要世界模型或生成模型，计算开销极低（一个20×20的栅格更新+简单统计），适合ARM CPU部署

2. **GRU双刃剑效应的系统分析与解决**：量化"隐式记忆（GRU）在不同POMDP场景下的正负效应"，并提出显式belief state作为替代方案——在保留记忆优势（窄门精确对齐）的同时避免其劣势（路径惯性导致无法回头）

3. **POMDP视角对U-trap失败的全新解释**：标准MDP视角下，U-trap失败被归因于"探索不足"（需要更强的探索奖励）；POMDP视角下，失败被归因于"信息不足"（机器人不知道自己在陷阱里）——两种解释导致完全不同的解决方案，我们通过实验对比两者

---

## 5. Related Work

**Paper:**

| 论文 | 关联 | 链接 |
|------|------|------|
| QMDP-Net, NeurIPS 2017 | 首个将POMDP结构嵌入神经网络的工作，但用于视觉栅格导航 | https://proceedings.neurips.cc/paper_files/paper/2017/file/e9412ee564384b987d086df32d4ce6b7-Paper.pdf |
| FORBES (Flow-based Recurrent Belief State), ICML 2022 | 用Normalizing Flow建模belief state，但计算开销大 | https://proceedings.mlr.press/v162/chen22q/chen22q.pdf |
| DreamerNav (DreamerV3 + Navigation), Frontiers 2025 | POMDP+世界模型导航，依赖深度相机，非LiDAR | http://frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1655171/full |
| DQN+GRU for LiDAR Navigation, arxiv 2021 | 用GRU近似POMDP，但未分析GRU的局限性 | https://ui.adsabs.harvard.edu/abs/2021arXiv211202954K/abstract |
| Spatially-Enhanced Recurrent Memory, arxiv 2025 | 空间增强记忆用于长距离导航，但未显式构建belief state | https://export.arxiv.org/abs/2506.05997 |

**Difference:**
- QMDP-Net/FORBES用复杂生成模型建模belief，我们用**轻量级占用记忆图**（无需训练额外生成模型）
- DQN+GRU只是隐式近似POMDP，我们**显式构建belief state**并分析其对策略的影响
- DreamerNav依赖深度相机+世界模型，我们是**纯LiDAR端到端方案**，适合资源受限平台
- 我们是**系统量化GRU在不同POMDP导航场景下双刃剑效应**的工作

---

## 6. Feasibility

| 项目 | 说明 |
|------|------|
| Data | IR-SIM / Isaac Sim在线生成 |
| Simulator | IR-SIM（当前），后续可迁移Isaac Sim |
| Code | 在CNNTD3基础上扩展状态空间（加占用记忆图分支），约200行新代码 |
| GPU | 没要求 |
| Difficulty | 3/5（占用记忆图的坐标变换和栅格更新有一定工程难度，但数学不复杂） |

---

## 7. Expected Contribution

- **理论贡献**：提出LiDAR导航中POMDP vs MDP建模的系统分析框架；揭示GRU隐式记忆的双刃剑效应机制（窄门有利、双U有害）；提出"信息不足"（POMDP视角）vs"探索不足"（MDP视角）对U-trap失败的两种竞争性解释，并通过实验区分两者
- **工程贡献**：轻量级Belief State模块（占用记忆图+轨迹编码），推理开销<0.5ms，可在ARM CPU上实时运行；即插即用，可集成到任意LiDAR RL导航框架
- **实验贡献**：首次在结构化困难场景（U-trap、双U、对称走廊、窄门）下系统对比MDP策略、GRU隐式记忆、显式Belief State三种方案的效果差异

---

## 8. Risks

1. **占用记忆图的累积误差**：机器人里程计有漂移，随时间推移栅格地图会越来越不准确。在仿真中不是问题（IR-SIM的里程计是理想的），但真机部署时需要处理。解决：限制记忆长度（只保留最近100步的观测），越老的信息权重越低

2. **状态空间增大导致训练更慢**：加了20×20栅格图意味着状态维度从185增加到185+400=585，网络参数增多，训练时间可能增加50-100%。解决：栅格图用小型CNN压缩到低维特征（如16维），和原始LiDAR特征拼接

3. **"信息不足" vs "探索不足"这两种解释可能难以实验区分**：如果Belief State方案和EWC方案都能解决U-trap，审稿人会问"到底是哪个起了作用"。解决：设计交叉消融实验（Belief State only / 探索奖励only / 两者结合）

4. **与简单GRU的对比可能不够公平**：GRU只用了10步历史，如果增加到50步或100步可能也能学到类似的belief信息。解决：系统扫描GRU的历史长度（5/10/20/50/100步），证明即使增加长度也无法弥补"隐式记忆"的结构性缺陷

---

## 9. Next Step

- [ ] 实现局部占用记忆图模块（坐标变换+栅格更新+egocentric平移旋转）
- [ ] 扩展CNNTD3网络结构，加入栅格图CNN分支
- [ ] 系统扫描GRU历史长度（5/10/20/50/100步），量化隐式记忆的天花板
- [ ] 对比实验：CNNTD3（MDP）vs RCPG（隐式POMDP）vs CNNTD3+Belief（显式POMDP）
- [ ] 在5个结构化困难场景系统评测
- [ ] 真机部署验证（占用记忆图在真实LiDAR噪声下的鲁棒性）

---

## 10. Reviewer Questions

- **Q1** 局部占用记忆图和SLAM有什么区别？为什么不直接用SLAM？
  → SLAM构建全局地图，计算开销大，且与"端到端RL"理念矛盾（引入了显式地图构建模块）。我们的占用记忆图是egocentric的、局部的（只覆盖10m×10m）、短期的（只保留最近的观测），本质上是一个"传感器记忆缓冲区"而非地图

- **Q2** 为什么不用Transformer代替GRU来处理历史观测？
  → Transformer确实可以更好地捕捉长程依赖，但计算开销显著高于我们的占用记忆图方案。我们的方案不需要注意力机制——栅格图本身就是一种空间注意力的显式实现

- **Q3** 20×20的栅格分辨率是否足够？在密集障碍物环境中会不会丢失细节？
  → 0.5m分辨率对于TurtleBot3（直径约0.2m）来说足够区分可通过/不可通过的间隙。更高分辨率（如0.1m，100×100栅格）可以作为消融实验

- **Q4** 你们的POMDP方案和EWC方案都能解决U-trap，如何区分两者的贡献？
  → 这正是我们的实验设计优势：EWC解决的是"训练过程中的遗忘问题"，POMDP解决的是"推理时的信息不足问题"。如果POMDP方案在**没有课程学习**的情况下也能提升U-trap SR，就证明信息不足是独立于遗忘的另一个失败原因

- **Q5** 真机上里程计漂移怎么处理？
  → 三种方案：(1)限制记忆长度，只保留最近100步；(2)用LiDAR scan matching做局部校正（不是完整SLAM）；(3)在训练时加入里程计噪声做domain randomization

- **Q6** 这个方法和World Model（如DreamerV3）有什么本质区别？
  → World Model学习环境的完整动力学模型（状态转移+奖励预测），计算开销大；我们的占用记忆图只做"观测累积"（把多帧LiDAR叠加到一张图上），不做任何预测，是一种"被动记忆"而非"主动建模"，计算开销低两个数量级
