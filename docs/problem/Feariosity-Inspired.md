# Research Topic:Feariosity-Inspired LiDAR RL Navigation for AMR

## 1. Motivation

为什么值得做？
- 自主移动机器人（AMR）的端到端RL导航面临两个核心矛盾：**安全性**（不能撞墙）和**探索性**（必须主动探索新区域才能逃出局部最小值）
- 现有RL导航要么偏向保守（不敢探索 → 卡在U形陷阱），要么偏向激进（敢探索 → 碰撞率升高）。这个矛盾在**结构化困难场景**（凹形陷阱、对称走廊）中尤其突出
- 神经科学研究表明，生物在探索中会同时激活**恐惧（Fear）**和**好奇心（Curiosity）**两种机制，并在两者之间动态平衡——这正是"Feariosity"的生物学基础（Hu et al., RA-L 2025）
- 现有Feariosity工作（Hu et al., 2025）针对**自动驾驶大场景（AGV/车辆）**，依赖**世界模型**，计算开销大，不适合资源受限的AMR（如TurtleBot3）
- 在**LiDAR端到端RL**的AMR场景中，轻量化Feariosity机制**尚无系统研究**
---

## 2. Core Idea

你打算怎么做？
核心思路：**不使用世界模型**，直接从LiDAR数据中实时提取恐惧信号和好奇心信号，动态调节探索行为。


**恐惧信号（Fear）**：
- 来源：LiDAR扫描的最小障碍物距离 $d_{min}$
- 含义：离障碍物越近 → 恐惧越强 → 抑制激进探索
- 实现：$f_{fear} = \exp(-\alpha \cdot d_{min})$，距离越小，恐惧值越大

**好奇心信号（Curiosity）**：
- 来源：与历史访问位置的距离
- 含义：到了没去过的地方 → 好奇心增强 → 鼓励继续探索
- 实现：$f_{curiosity} = \min_{p \in \mathcal{P}} \|x_t - p\|$，与历史轨迹距离越大，好奇心越强

**Feariosity动态调节**：
$$r_{feariosity} = \beta \cdot f_{curiosity} - \gamma \cdot f_{fear}$$

其中 $\beta$、$\gamma$ 可以固定，也可以根据训练阶段动态调节（课程学习早期 $\beta$ 大鼓励探索，后期 $\gamma$ 大保证安全）

**与TD3结合**：
$$\mathcal{L}_{total} = \mathcal{L}_{TD3} + r_{feariosity}$$

这个设计**不需要世界模型**，直接在原始LiDAR观测上计算，推理开销接近零。
---

## 3. Research Gap

别人没有解决什么？
- **Feariosity 思想在 LiDAR AMR 上的适用性未知**：[Hu et al. 2025](https://arxiv.org/html/2510.10960v1) 的Feariosity模型针对自动驾驶，使用世界模型，计算开销大；在TurtleBot3等资源受限AMR上的适用性未被验证
- **轻量级Feariosity实现**：现有Feariosity依赖世界模型来建模"威胁感知"，我们直接从LiDAR的 $d_{min}$ 提取恐惧信号，无需世界模型，降低了计算复杂度
- **距离依赖的 fear 信号 vs. 纯好奇心探索**：现有 AMR RL 导航中的探索奖励（ICM、RND、state count）只建模好奇心，没有显式的"靠近障碍物时抑制探索"机制。将两者结合是否优于单独使用，需要系统实验回答
- **探索奖励强度的敏感性**：已知 intrinsic reward 的缩放系数对性能影响大且需要手动调参（Burda et al., 2019; Taiga et al., 2020）。我们的预实验观察到类似阈值效应（0.3 有效，0.15 无效），这提示状态依赖的动态调节可能比固定系数更鲁棒

## 4. Novelty

真正的创新点是什么？

1. **无世界模型的轻量级Feariosity**：直接从LiDAR $d_{min}$ 提取恐惧信号，从位置历史提取好奇心信号，计算开销接近零，适合资源受限的AMR平台（TurtleBot3 Burger，ARM CPU，推理延迟<2ms）

2. **Feariosity框架对探索奖励阈值效应的理论解释**：用Fear-Curiosity动态平衡解释为什么探索奖励存在最低生效阈值（非线性相变）——只有当好奇心信号强度超过恐惧信号的抑制作用时，探索行为才会被真正激活，这比单纯的"调参发现"提供了更深的理论洞察

3. **Feariosity与课程学习的结合**：在课程学习的不同阶段动态调节 $\beta/\gamma$ 比值——早期 $\beta > \gamma$（好奇心主导，积极探索新场景），后期 $\gamma$ 增加（恐惧约束增强，安全优先），使Feariosity与课程学习协同工作

---

## 5. Related Work

| 论文 | 关联 | 链接 |
|------|------|------|
| Hu et al., IEEE RA-L 2025，"Feariosity"-Guided RL | 方法基础，直接前驱工作 | https://ieeexplore.ieee.org/document/11027413 |
| He et al., IEEE TPAMI 2023, Fear-Neuro-Inspired RL | 恐惧模型的神经科学基础 | https://ieeexplore.ieee.org/document/10273631 |
| Pathak et al., ICML 2017, ICM | 好奇心驱动探索经典方法 | https://ieeexplore.ieee.org/document/8014804 |
| Burda et al., ICLR 2019, RND | 随机网络蒸馏好奇心 | https://arxiv.org/abs/1810.12894 |
| Yang et al. 2024, ICM + TD3 for AMR | 好奇心+TD3在移动机器人上的应用 | https://journals.sagepub.com/doi/10.1177/17298806241292893 |
| Tai et al., IROS 2017 | RL导航基线 | https://ieeexplore.ieee.org/document/8202134 |


**Difference:**
- Hu et al. 2025用世界模型实现Feariosity，针对车辆平台；**我们不用世界模型，针对AMR**
- ICM/RND用预测误差衡量好奇心（计算开销大）；**我们用位置历史距离（无需额外网络）**
- Yang et al. 2024只有好奇心，没有恐惧约束；**我们显式建模Fear-Curiosity动态平衡**

---

## 6. Feasibility

| 项目 | 说明 |
|------|------|
| Data | IR-SIM / Isaac Sim 在线生成，无需额外数据集 |
| Simulator | IR-SIM（当前，2D LiDAR），后续迁移Isaac Sim |# Research Topic:Feariosity-Inspired LiDAR RL Navigation for AMR
| Code | 在现有CNNTD3基础上修改exploration bonus为Feariosity公式，约80行新代码 |
| GPU | 都可以 |
| Difficulty | 2/5（公式简单，代码改动小，难点在实验分析和理论解释） |

### 关键消融实验

**实验 A：β/γ 比值扫描**
- 固定 β+γ=1，扫描 β/γ ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 2.0, 5.0, 10.0}
- 在 S3 (U-Trap) 上测试，画 SR 和 CR 随 β/γ 的变化曲线
- 预期：存在一个最优区间（过低 = 保守，过高 = 碰撞）

**实验 B：阈值效应验证**
- 静态探索奖励强度扫描：{0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5}
- 同时记录训练过程中 $f_{fear}$ 和 $f_{curiosity}$ 的均值
- 分析：阈值点是否对应 $\beta \cdot \bar{f}_{curiosity} \approx \gamma \cdot \bar{f}_{fear}$ 的临界条件

**实验 C：动态 vs. 固定 β/γ**
- 对比固定最优 β/γ vs. 课程学习动态调节
- 如果动态没有显著优势，诚实报告

**实验 D：计算开销对比**
- 测量每个方法的单步推理时间（在 TurtleBot3 ARM CPU 上）
- ICM vs. RND vs. FearCuriosity reward
- 轻量化方案的核心卖点，必须有硬数据

---

## 7. Expected Contribution

- **方法贡献**：提出受 Feariosity 启发的状态依赖探索奖励，仅依赖 LiDAR $d_{min}$ 和位置历史，无需额外网络或世界模型，可集成到任意 LiDAR RL 导航框架
- **实验贡献**：在结构化困难场景（U-trap、Double-U、对称走廊）下系统评估 fear-curiosity 机制；消融实验分析 Fear/Curiosity 各自贡献；与 ICM/RND/静态探索奖励的对比；计算开销的量化对比
- **可解释性贡献**：基于 fear-curiosity 视角，对探索奖励阈值效应提供机制性解释（hypothesis + empirical evidence）
---

## 8. Risks
| 风险 | 概率 | 应对 |
|------|------|------|
| FearCuriosity 在 U-trap 上没有显著优于静态 0.3 奖励 | 中高 | 如果发生，论文转向"负面结果 + 分析为什么"，仍有发表价值 |
| β/γ 最优值高度场景依赖，无法泛化 | 中 | 诚实报告，提出自适应调节作为 future work |
| 审稿人认为贡献量不够 | 中 | 加强真机实验 + 计算开销对比，把"轻量化 + 真机可部署"做扎实 |
| 审稿人认为只是 reward shaping 的变体 | 高 | 不回避这一点，而是论证：(1) 这个 shaping 有神经科学动机，(2) 系统实验证明了它在困难场景中的有效性，(3) 提供了对阈值效应的解释视角 |
| Fear 信号过强抑制探索 | 中 | 系统扫描 β/γ 比值，寻找最优平衡点 |
| 好奇心信号在课程学习场景切换时失效 | 低 | 切换场景时用滑动窗口保留部分历史 |

---

## 9. Next Step

- [ ] 将现有的 `get_exploration_bonus()` 改写为Feariosity公式（加入fear信号）
- [ ] 实现动态 $\beta/\gamma$ 调节（课程阶段感知）
- [ ] 系统扫描 $\beta/\gamma$ 比值（消融实验），找最优平衡点
- [ ] 补充探索奖励强度扫描实验（0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4），完善阈值效应数据
- [ ] 与ICM/RND实现对比（验证轻量化设计的效率优势）
- [ ] 在4个结构化困难场景（S1-S5）系统评测
- [ ] 迁移至Isaac Sim提高训练速度
- [ ] TurtleBot3真机部署验证

## 10. Reviewer Questions

- **Q1** 和Hu et al. 2025的Feariosity有什么本质区别，不只是去掉了世界模型吧？
  → 本质区别有两点：(1) 我们针对LiDAR AMR的资源约束设计了无网络的信号提取方案；(2) 我们用Feariosity框架解释了探索奖励阈值效应这一新发现，提供了比原论文更深的理论洞察

- **Q2** 为什么不用ICM（预测误差好奇心）？LiDAR位置历史作为好奇心信号是否足够？
  → ICM需要额外的预测网络，计算开销大；位置历史在导航场景中有直接的物理意义（是否探索过这个区域），对LiDAR导航任务来说更直观、更鲁棒

- **Q3** $d_{min}$ 作为恐惧信号是否过于简单？真实环境中 $d_{min}$ 噪声很大？
  → 简单是设计目标（轻量化）；真实LiDAR（RPLIDAR C1）的噪声在验证实验中量化，我们会分析噪声对恐惧信号的影响并提供鲁棒性分析

- **Q4** 与直接调节探索奖励强度（你们做过的实验）相比，Feariosity有什么优势？
  → 静态探索奖励强度是固定的（0.3），Feariosity是动态的——靠近障碍物时自动降低探索强度，远离障碍物时自动提高，不需要手动调参；这正是对阈值效应的"解决方案"而非"发现"

- **Q5** 为什么在结构化困难场景（U-trap）而不是开放场景验证？
  → 开放场景中探索-安全矛盾不突出（绕开一个障碍物即可），U-trap场景才能真正考验Feariosity：既要克服恐惧进入U形开口，又要保持足够探索性发现出口
