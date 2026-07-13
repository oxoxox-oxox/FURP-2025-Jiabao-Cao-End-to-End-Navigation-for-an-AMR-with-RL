"""
DUNE 延迟感知重训练（预处理NeuPAN以实现公平对比）

原理：
  原始DUNE训练：输入=精确点坐标(x,y)，标签=该点到机器人形状的真实距离
  延迟训练：   输入=偏移后的点坐标（模拟"用t=10s的数据在t=12s使用"造成的观测偏移），
              标签=真实位置的距离（不变）
  → 网络学会：从过时/偏移的观测中推断真实距离，对延迟造成的输入误差鲁棒

延迟→偏移的换算：
  观测延迟d秒内，机器人（和障碍物）相对移动了 v×d 米，
  所以点坐标的偏移量 = 速度采样 × 固定延迟
  例：延迟0.5s，速度0-4m/s → 偏移0-2m，方向随机

用法：
  cd ~/NeuPAN/example/dune_train && conda activate neupan
  python dune_train_delay.py                     # 固定500ms延迟
  python dune_train_delay.py --delay-s 0.3       # 固定300ms
  python dune_train_delay.py --random-delay      # 随机延迟0~delay-s（第二阶段用）

训练完成后，模型保存在 example/model/ 下的新目录，
把 planner yaml 里的 dune_checkpoint 指向新模型即可测试。
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, '/home/ubuntu22/NeuPAN')
os.chdir('/home/ubuntu22/NeuPAN/example/dune_train')

parser = argparse.ArgumentParser()
parser.add_argument("--delay-s", type=float, default=0.5,
                    help="固定延迟秒数（默认0.5s=500ms）")
parser.add_argument("--random-delay", action="store_true",
                    help="随机延迟：每个样本 d ~ Uniform(0, delay_s)。默认关闭=固定延迟")
parser.add_argument("--max-speed", type=float, default=4.0,
                    help="偏移速度上限 m/s（对应planner的ref_speed=4）")
parser.add_argument("--config", type=str, default="dune_train_diff.yaml",
                    help="DUNE训练配置yaml")
args = parser.parse_args()

DELAY_S = args.delay_s
RANDOM_DELAY = args.random_delay
MAX_SPEED = args.max_speed

print(f"{'='*60}")
print(f"DUNE 延迟感知重训练")
print(f"  延迟 = {DELAY_S}s ({'随机 0~' + str(DELAY_S) + 's' if RANDOM_DELAY else '固定'})")
print(f"  速度上限 = {MAX_SPEED} m/s")
print(f"  最大点偏移 = {MAX_SPEED * DELAY_S:.2f} m")
print(f"{'='*60}\n")

# ============================================================
# Monkey-patch: 替换 DUNETrain.generate_data_set
# 不修改NeuPAN库文件本身
# ============================================================
from neupan.blocks.dune_train import DUNETrain, PointDataset
from neupan.configuration import np_to_tensor, value_to_tensor

_original_generate = DUNETrain.generate_data_set

def generate_data_set_with_delay(self, data_size=10000, data_range=[-50, -50, 50, 50]):
    """延迟版数据生成：
    - 真实点 p_true: 随机采样，标签 = convex_solve(p_true)（真实距离）
    - 输入点 p_obs = p_true + 延迟偏移
      偏移 = 随机方向 × (速度采样 × 延迟)
    """
    input_data, label_data, distance_data = [], [], []

    rand_p = np.random.uniform(
        low=data_range[:2], high=data_range[2:], size=(data_size, 2)
    )

    for i in range(data_size):
        p_true = rand_p[i].reshape(2, 1)

        # 标签：真实位置的距离（凸优化求解）
        distance_value, mu_value = self.prob_solve(p_true)

        # 延迟偏移
        if RANDOM_DELAY:
            d = np.random.uniform(0.0, DELAY_S)
        else:
            d = DELAY_S
        speed = np.random.uniform(0.0, MAX_SPEED)
        angle = np.random.uniform(-np.pi, np.pi)
        offset = np.array([[speed * d * np.cos(angle)],
                           [speed * d * np.sin(angle)]])

        p_obs = p_true + offset  # 网络看到的是偏移后的点

        input_data.append(np_to_tensor(p_obs))          # 输入：延迟观测
        label_data.append(np_to_tensor(mu_value))        # 标签：真实mu
        distance_data.append(value_to_tensor(distance_value))  # 标签：真实距离

        if (i + 1) % 10000 == 0:
            print(f"  数据生成: {i+1}/{data_size}")

    return PointDataset(input_data, label_data, distance_data)

# 应用patch
DUNETrain.generate_data_set = generate_data_set_with_delay
print("[INFO] DUNETrain.generate_data_set 已替换为延迟版本\n")

# ============================================================
# 修改模型保存名（避免覆盖原模型）
# ============================================================
import yaml
with open(args.config) as f:
    cfg = yaml.safe_load(f)

# 给模型起新名字
suffix = f"delay{int(DELAY_S*1000)}ms" + ("_rand" if RANDOM_DELAY else "_fixed")
if "train" not in cfg:
    cfg["train"] = {}
cfg["train"]["model_name"] = f"diff_robot_{suffix}"

tmp_config = f"_tmp_dune_train_{suffix}.yaml"
with open(tmp_config, "w") as f:
    yaml.dump(cfg, f)

print(f"[INFO] 模型将保存为: example/model/diff_robot_{suffix}/\n")

# ============================================================
# 启动训练（走NeuPAN原生流程）
# ============================================================
from neupan import neupan

neupan_planner = neupan.init_from_yaml(tmp_config)
neupan_planner.train_dune()

# 清理临时配置
os.remove(tmp_config)

print(f"\n{'='*60}")
print(f"训练完成！")
print(f"下一步：修改 planner_turn_simple.yaml 中的 dune_checkpoint 为：")
print(f"  'example/model/diff_robot_{suffix}/model_5000.pth'")
print(f"然后跑延迟测试对比原模型 vs 延迟训练模型")
print(f"{'='*60}")