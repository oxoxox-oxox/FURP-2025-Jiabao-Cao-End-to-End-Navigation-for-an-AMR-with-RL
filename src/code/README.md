# Code map / 代码导览

## Final method / 最终方法

**STPS v2** lives in `scripts/test/eval_unified.py`, especially `run_stps_v2` and `detect_oscillation`. The file also loads checkpoints and launches a full benchmark at module level; importing it directly starts that work. `tools/stps_source.py` reads the original definitions with Python AST without executing checkpoint loading or duplicating the algorithm.

**最终方法为 v2**，见上述函数。原脚本顶层会加载模型并运行评估，不能当普通模块直接导入。根目录辅助工具以 AST 读取原函数及常量，不执行顶层评估，也不复制算法。

`scripts/eval/script3_stps_v3.py` is an exploratory extension with goal-distance detection, progressive 120/180/240-step escapes and cooldown. It is not the final method and is not a prerequisite for v2.

v3 增加目标距离检测、渐进脱困及冷却期，属探索扩展，不是最终方法，也不是 v2 的运行前提。

## Observation → policy → simulator / 观测 → 策略 → 仿真

1. `robot_nav/worlds/` YAML sets geometry, starts, goals, differential-drive limits and noisy 180-beam LiDAR.
2. `robot_nav/sim/sim.py` exposes reset/step, distance, heading, collision, arrival and reward.
3. `robot_nav/models/cnntd3.py` normalizes LiDAR by 7 m and distance by 10 m, then appends heading cosine/sine and normalized previous velocities. Collision/goal flags determine termination; they are **not** part of the 185-dimensional actor input.
4. The CNN actor emits normalized actions. The v2 evaluator maps them to linear velocity `[0,0.5]` and angular velocity `[-1,1]`, selects the active policy and steps the simulator.
5. Episodes terminate on goal/collision or 500-step timeout; 12 perturbed starts per seed and seeds `[42,123,2026]` produce scenario success rates.

YAML 定义场景与雷达，仿真接口产生观测，CNN 将雷达和目标/上次动作编码后输出速度，切换器选择当前策略。成功、碰撞或超时结束回合。185 维状态不含碰撞和到达标志，两者只用于终止判断。

## Other preserved modules / 其他保留模块

- `models/`: TD3, DDPG, PPO, recurrent RCPG, SAC, MARL and hard-coded baselines. Their presence does not imply all have been verified.
- `replay_buffer.py`, `utils.py`: replay, offline pretraining and bound calculations. `assets/data.yml` is an existing ~44 MB experience dataset, retained unchanged; the smoke workflow does not load it.
- `scripts/train/rl_train_v6_anneal.py`: trains with exploration annealing and saves `CNNTD3_improved_best`/epoch checkpoints.
- `scripts/train/rl_train_v7_finetune.py`: actually loads `CNNTD3_improved_best` despite the older docstring referring to v5; the starting checkpoint is absent.
- `scripts/eval/`: diagnostic/sensitivity experiments. The sensitivity script uses a 60-step escape setting, differing from the final unified v2 controller's initial 120 steps.

其他模型、回放、离线预训练和探索脚本均保留。数据集约 44 MB，冒烟流程不读取。v7 的真实起点以代码中的 `improved_best` 为准；敏感性脚本的 60 步脱困设置不能直接等同于最终 120 步控制器。

For exact commands, missing checkpoint names and use restrictions, see [reproduction](../../docs/REPRODUCIBILITY.md) and [rights](../../RIGHTS.md). / 完整命令、资源缺项与使用条件见链接。
