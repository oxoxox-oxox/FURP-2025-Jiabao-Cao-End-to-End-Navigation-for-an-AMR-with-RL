[English](README.md) · [简体中文](README.zh-CN.md)

# STPS：让机器人脱困，同时保留精确导航能力

**FURP 团队项目：面向移动机器人的端到端强化学习导航。**

**[研究海报 PDF](FURP_Showcase.pdf) · [演示文稿 PPTX](FURP_Showcase.pptx)**

<a href="FURP_Showcase.pdf"><img src="docs/assets/stps-poster.png" alt="STPS 研究海报" width="600"></a>

## 价值 → 贡献 → 方法

持续朝目标前进的机器人可能困在凹形障碍后方。探索有助于脱困，却可能破坏通过窄门所需的精度。**停滞触发策略切换（STPS）**在运行时组合两种已训练的互补策略，处理这一冲突。

团队贡献包括停滞/振荡检测器、精确导航与探索脱困的切换控制器，以及四类困难场景下条件一致的对比评估。仓库包含 CNN-TD3 模型、仿真接口、训练/评估脚本与逐种子结果汇总，体现策略组合、机器人连续控制和对比实验能力。


**最终方法为 STPS v2**。20 个位置的窗口中位移小于 0.15 m，或 12 个位置中出现至少 5 次方向反转时触发脱困；达到最短脱困时长且位移超过 0.5 m 后返回，并设有超时退出。两种策略参数冻结，切换机制无需额外训练。V3 增加目标距离检测与冷却期，属于探索版本，不作为最终成果。

## 成果与证据

| 方法 | U-trap | Double-U | 窄门 | 走廊 | 困难场景平均成功率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN-TD3 基线 | 0% | 69±4% | 100% | 100% | 67.36% |
| **STPS v2** | **75±7%** | **100%** | **100%** | **100%** | **93.75%** |

依据[存档逐种子结果](src/results/final/unified_comparison.json)：每个困难场景为 3 个种子 × 12 个扰动起点，± 为种子成功率的总体标准差。平均提升 **26.39 个百分点**；海报将均值取整为 94% 和 67%。[最终表格](src/results/final/final_results_table.md)另报告 100 回合标准环境成功率 88% 与 87%，但该项缺少逐回合日志。

这些数值是**从存档汇总重算**，并非重新运行训练后策略。仓库缺少策略权重。本次未独立验证海报中的计算开销及实机部署说法。[证据与限制](docs/REPRODUCIBILITY.md)说明了实际可复现范围。

## 快速开始

仅用 Python 重算存档指标，不需要模型、GPU 或账号：

```sh
git clone https://github.com/oxoxox-oxox/FURP-2025-Jiabao-Cao-End-to-End-Navigation-for-an-AMR-with-RL.git
cd FURP-2025-Jiabao-Cao-End-to-End-Navigation-for-an-AMR-with-RL
python tools/summarize_results.py
```

预期输出包含 `CNNTD3_baseline ... 67.36%`、`NeuPAN ... 0.00%`、`STPS_v2 ... 93.75%`。NeuPAN 数值仅适用于存档的紧凑场景与配置，不能推广为该方法普遍失效。

仿真、切换逻辑和模型接口检查使用 Python 3.12：

```sh
python -m venv .venv
# Windows PowerShell：
.venv\Scripts\Activate.ps1
# Linux shell 则使用：source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -X utf8 tools/verify_project.py
```

成功时末尾显示 `VERIFICATION PASSED`。真实 IR-SIM 冒烟测试在四类场景调用原切换函数，使用明确标注的静止/旋转合成策略：每场景主策略调用 19 次、脱困策略 41 次，以超时结束。另一项检查使用随机初始权重运行原 CNN-TD3 actor/critic。**两者均不衡量学习后的导航性能。**生成文件保存在被忽略的 `artifacts/`。

安装需联网，CPU PyTorch 需要较大磁盘空间；验证本身离线运行，耗时数秒至数十秒，无付费服务，也无需凭据。固定依赖、完整命令和资源缺失检查见[复现说明](docs/REPRODUCIBILITY.md)。

## 结构、开发与状态

`src/code/robot_nav/worlds/` 定义场景，`sim/` 将雷达、目标几何与上一步动作组成观测；`models/cnntd3.py` 编码 **180 束雷达 + 5 个目标/动作数值 = 185 维输入**，输出两个速度动作；`scripts/test/eval_unified.py` 选择精确/探索策略、执行回合并汇总成功率。`src/results/final/` 保存原始成果，`docs/` 保留研究历史，详见[源码导览](src/code/README.md)。

仓库为研究快照。完整 STPS 策略推理缺少 12 个权重文件；`python tools/check_resources.py --strict` 列出缺项并返回状态码 2。历史训练和硬件流程未验证。核心代码与数据保持原样，见[验证记录](docs/VALIDATION.md)。

运行 `python -X utf8 -m unittest discover -s tests -v` 执行测试；CI 还验证仿真器与 CPU 模型。通过 [GitHub Issues](https://github.com/oxoxox-oxox/FURP-2025-Jiabao-Cao-End-to-End-Navigation-for-an-AMR-with-RL/issues)反馈命令与环境，勿包含秘密信息。复用或修改前须取得许可。

**保留所有权利，不授予开源许可证。**公开仓库供通过平台查看，其他使用、再分发或修改须取得相关权利人许可；团队作者、导师及外部软件的署名与权利保留，详见[权利与致谢](RIGHTS.md)。
