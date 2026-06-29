## 已训练模型

| 模型 | 文件 | 标准SR | U-trap SR |
|------|------|--------|-----------|
| CNNTD3 | checkpoint/CNNTD3_actor.pth | 92% | 0% |
| RCPG | checkpoint/RCPG_actor.pth | 88% | 0% |
| CNNTD3_improved | checkpoint/CNNTD3_improved_actor.pth | 90% | 100% |
| ATD3 | checkpoint/ATD3_actor.pth | 训练中 | 待测 |

## 训练

```bash
# 标准训练
cd robot_nav && python rl_train.py

# 课程学习+探索奖励（推荐）
cd robot_nav && python rl_train_improved.py

# ATD3（Attention架构，100点雷达）
cd robot_nav && python rl_train_atd3.py
```

## 测试

```bash
# 全场景benchmark（三个模型）
python benchmark_all_models.py

# NeuPAN走廊对比
python test_neupan_corridor.py
```

## 场景说明

| 场景 | 文件 | 挑战 |
|------|------|------|
| 标准 | robot_world.yaml | 随机障碍 |
| S1 U形陷阱 | u_trap_world.yaml | 凹形陷阱 |
| S2 双U | double_u_world.yaml | 双重陷阱 |
| S3 窄门 | narrow_door_world.yaml | 精确对齐 |
| S5 走廊 | symmetric_corridor_world.yaml | 对称死锁 |
| 走廊测试 | corridor_world_20.yaml | 长走廊避障 |

## NeuPAN对比结果

| 场景 | CNNTD3 | Improved | NeuPAN |
|------|--------|----------|--------|
| 标准 | 92% | 90% | 0% |
| U形 | 0% | 100% | 0% |
| 双U | 33% | 33% | 0% |
| 窄门 | 5% | 0% | 0% |
| 走廊 | 83% | 100% | 0% |
