
## 一、端到端导航领域整体局限性（大范围背景）

| 序号 | 问题 | 简述 |
|---|---|---|---|
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
|---|---|---|---|---|
| 1 | CPU-bound optimization: The NRMP layer uses cvxpy which does not support GPU acceleration | NRMP层用cvxpy，不支持GPU加速，推理速度受CPU限制 | 
| 2 | Supported kinematics: Currently limited to differential drive, Ackermann, and omnidirectional robots | 仅支持差速/阿克曼/全向轮三种运动学 | 
| 3 | Convex robot geometry: The DUNE model assumes convex robot shapes | DUNE假设机器人是凸形，非凸需凸包近似 | 
| 4 | Parameter tuning: Performance in specific environments may require tuning the `adjust` parameters | 特定环境下需手动调参 |


---

# 方案一、NeuPAN腿足扩展

不修改NeuPAN的DUNE/NRMP核心数学公式，只在"用哪个训练好的形状模型"这一层做文章：训练两个独立的DUNE模型（对应两种典型步态形状），外层加一个简单的切换逻辑。
NeuPAN 官方明确表示腿足等其他运动学需要自己改约束，且经检索，**目前没有公开工作把 NeuPAN 的 DUNE+NRMP 框架应用到腿足机器人的底盘运动规划上**——已有的腿足避障工作都走的是经典 MPC 或纯 RL 路线，没有人用 NeuPAN 这种"点云直接映射隐式距离特征+可微凸优化"的范式做腿足导航。
将 NeuPAN 的端到端模型驱动避障框架扩展到腿足机器人底盘运动学约束，对比纯 RL / 纯 MPC 方法在可解释性、训练效率、sim-to-real 泛化上的优势。

| 步骤 | 内容 | 预计时间 |
|---|---|---|
| 1 | 量取机械狗两种典型形状的尺寸参数（双脚支撑/单脚迈步） | 1天 |
| 2 | 用NeuPAN官方`example/dune_train`脚本，分别训练形状A、B两个DUNE模型 | 2-3天 |
| 3 | 编写"何时用A、何时用B"的切换逻辑（先用固定时间节奏的朴素版本） | 3-5天 |
| 4 | 训练baseline（取A、B并集的保守包络形状）模型 | 1天 |
| 5 | 在ir-sim设计窄道测试场景，跑对照实验（动态切换 vs 保守包络） | 1周 |
| 6 | 整理数据、调参、补实验、写报告 | 1周 |
