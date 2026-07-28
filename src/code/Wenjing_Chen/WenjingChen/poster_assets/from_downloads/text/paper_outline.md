# Paper Outline

## Title (方向)
"Curriculum-Enhanced Deep Reinforcement Learning for Navigation
in Structured Trap Environments"

或中文：
"基于课程学习增强的深度强化学习导航：结构化陷阱环境中的系统评估与改进"

---

## Abstract (~200 words)

- 问题：端到端 RL 导航在标准环境表现优秀（SR>90%），
  但在凹形陷阱等结构化困难场景中完全失败（SR=0%）
- 发现：GRU 记忆对精确导航有效（+85.7%），但对陷阱反而有害（-67%）
- 方法：课程学习 + 探索奖励
- 结果：U 形陷阱 SR 从 0% 提升到 100%，推理速度比 NeuPAN 快 15 倍
- 意义：训练分布比网络架构更重要

---

## 1. Introduction (~1 page)

### 1.1 背景
- 端到端 RL 导航是 mapless navigation 的主流方向
- 现有方法（DRL-based）在标准随机障碍环境效果好
- 但真实环境存在凹形障碍物、死路、对称走廊等结构化难点

### 1.2 问题
- 现有评测主要在随机障碍场景，无法暴露结构化失败模式
- 缺乏系统性的困难场景 benchmark
- 加记忆（LSTM/GRU）是否真的能解决这些问题？

### 1.3 贡献（3-4 点）
1. 设计了 4 类结构化困难场景 benchmark（U-trap, Double-U, Narrow-door, Symmetric-corridor）
2. 系统评测了 3 种方法（CNNTD3, RCPG, NeuPAN），发现 GRU 记忆是双刃剑
3. 提出课程学习 + 探索奖励的改进方法，U-trap SR 从 0% → 100%
4. 推理速度对比：RL 方法比 NeuPAN 快 15 倍（GPU）

---

## 2. Related Work (~1 page)

### 2.1 DRL for Mapless Navigation
- TD3, SAC, PPO 用于 LiDAR-based 导航
- CNN 处理 LiDAR 数据
- 引用: DRL-robot-navigation 项目, Tai et al., Pfeiffer et al.

### 2.2 Memory-Augmented Navigation
- LSTM/GRU 加入 RL 导航
- 现有假设：记忆能帮助 → 我们的发现：不一定
- 引用: LP-TD3 (2024), SRU (2025)

### 2.3 Local Minima and Trap Escape
- Bug 算法（经典方法）
- APF + wall-following（Kim et al. 2024）
- 环境预测（DreamFlow 2026）
- 引用: Kim et al. 2024, Miranda et al. 2024, DreamFlow 2026

### 2.4 NeuPAN
- Model-based neural planner with MPC
- 优势：精确避障。劣势：无法处理凹形陷阱
- 引用: NeuPAN (TRO 2025)

---

## 3. Method (~2 pages)

### 3.1 Problem Formulation
- 状态空间：180-beam LiDAR + goal direction (distance, cos, sin) + action
- 动作空间：线速度 + 角速度
- 奖励函数：goal=+100, collision=-100, 原始 shaping

### 3.2 Baseline: CNNTD3
- 1D CNN 处理 LiDAR → FC layers → TD3 actor-critic
- 网络结构图
- 训练参数：60 epochs × 70 episodes

### 3.3 Memory Baseline: RCPG
- GRU 替代 CNN，处理 10 步历史状态序列
- 和 CNNTD3 同样的训练环境和参数

### 3.4 Proposed Improvement
#### 3.4.1 Curriculum Learning
- 训练前期：100% 标准环境
- 训练中后期：逐步混入困难场景（概率从 0 增加到 30%）
- 困难场景包括：U-trap, U-shape, U-shape-hard

#### 3.4.2 Exploration Reward
- 探索新区域奖励：距离历史轨迹 >0.5m 时 +0.15
- 原地打转惩罚：连续不动时 -0.1
- 停滞惩罚：连续 15 步不动时 -0.3

---

## 4. Experimental Setup (~1 page)

### 4.1 Simulation Environment
- IR-SIM 仿真器
- Robot: diff drive, radius=0.2m, 180-beam LiDAR (range 7m, 180°)

### 4.2 Hard Scenario Benchmark
- S1 U-trap: agent 在 U 形内部，goal 在外部
  [图：U-trap 场景示意图]
- S2 Double-U: 两个 U 形陷阱对称分布
- S3 Narrow-door: 不同门宽（0.45m ~ 1.0m）
- S5 Symmetric-corridor: 对称走廊，不同初始朝向

### 4.3 Evaluation Metrics
- Success Rate (SR)
- Collision Rate (CR)
- Timeout Rate (TR)
- Inference time (ms/step)

### 4.4 Compared Methods
| Method | Architecture | Memory | Training |
|--------|-------------|--------|----------|
| CNNTD3 | CNN+TD3 | None | Standard env |
| RCPG | GRU+TD3 | GRU (10 steps) | Standard env |
| NeuPAN | MPC | None | No training |
| Ours | CNN+TD3 | None | Curriculum + exploration reward |

---

## 5. Results (~2 pages)

### 5.1 Standard Environment Performance
- 表格：4 个方法的 SR, CR, training time
- TensorBoard 训练曲线对比图

### 5.2 Hard Scenario Comparison
| Scenario | CNNTD3 | RCPG | NeuPAN | Ours |
|----------|--------|------|--------|------|
| S1 U-trap | 0% | 0% | 0% | **100%** |
| S2 Double-U | 33% | 0% | 0% | 67% |
| S3 Narrow-door | 5% | 91% | 0% | 10% |
| S5 Corridor | 83% | 100% | 0% | 100% |

[图：各场景的仿真截图]

### 5.3 GRU Memory Analysis
- 表格：RCPG vs CNNTD3 逐场景对比
- 发现：GRU 在精确导航 +85.7%，但在陷阱场景 -67%
- 原因分析：GRU 增加路径持续性

### 5.4 Ablation Study
| Method | Curriculum | Exploration Reward | S1 SR | Standard SR |
|--------|-----------|-------------------|-------|------------|
| CNNTD3 | No | No | 0% | 92% |
| + CL only | Yes | No | 0% | 81% |
| + Both | Yes | Yes | 100% | 78% |

- 结论：探索奖励是解决 U-trap 的关键因素

### 5.5 Inference Speed Comparison
| Method | CPU (ms) | GPU (ms) |
|--------|----------|----------|
| CNNTD3 | 7.5 | **0.38** |
| NeuPAN | 5.3 | 5.9 |

- GPU 上 CNNTD3 比 NeuPAN 快 15 倍
- 两者在 CPU 上速度相当，都满足 10Hz 实时要求

---

## 6. Discussion (~0.5 page)

### 6.1 Trade-off
- 标准环境 SR 下降（92% → 78%）是课程学习的代价
- 可通过增加训练 epoch 缓解

### 6.2 Limitations
- 只在 2D 仿真中验证，未在真机测试
- 困难场景设计较简单（只有 4 类）
- 探索奖励参数手动调节，未做系统搜索

### 6.3 Why Training Distribution Matters More Than Architecture
- RCPG 有记忆但仍然失败 → 架构不是关键
- 课程学习改变训练分布就能解决 → 数据是关键
- 这与近年来"data-centric AI"的趋势一致

---

## 7. Conclusion (~0.5 page)

- 构建了 4 类困难场景 benchmark
- 发现 GRU 记忆是双刃剑
- 课程学习 + 探索奖励解决了 U-trap（0% → 100%）
- 训练分布比网络架构更重要
- Future work: 真机部署, 更多困难场景类型, 自适应课程学习

---

## References (~20-30 篇)

核心引用：
- TD3: Fujimoto et al. 2018
- NeuPAN: TRO 2025
- DRL navigation: Tai et al., Pfeiffer et al.
- Curriculum learning: Bengio et al. 2009
- Reward shaping: Ng et al. 1999
- Local minima: Kim et al. 2024, DreamFlow 2026
- IR-SIM: 仿真器文档

---

## 需要的图表清单

1. 系统架构图（CNNTD3 网络结构）
2. 4 个困难场景的示意图
3. TensorBoard 训练曲线（5 个模型对比）
4. 困难场景仿真截图（成功 + 失败案例）
5. 三方/四方对比表格
6. 消融实验表格
7. 推理速度对比表格
8. 窄门 SR vs 门宽曲线
