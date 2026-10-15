# Reproduction and limits / 复现与限制

## What the checkout supports / 当前可执行范围

| Level / 层级 | Entry / 入口 | Meaning / 含义 |
| --- | --- | --- |
| Archived result arithmetic / 存档核算 | `python tools/summarize_results.py` | Recompute seed means/stds; no new episodes / 重算种子均值与标准差，不运行新回合 |
| Controller regression / 控制器回归 | `python -X utf8 -m unittest discover -s tests -v` | Exercise original v2 branches with controlled trajectories / 使用受控轨迹验证原 v2 分支 |
| Real simulation / 真实仿真 | `python -X utf8 tools/smoke_stps.py` | Original STPS function in four worlds, synthetic policies / 原 STPS 函数与四类场景，使用合成策略 |
| Model interface / 模型接口 | `python -X utf8 tools/smoke_model.py` | Real LiDAR → original CPU actor/critics → simulator, random weights / 雷达至原模型及仿真，随机权重 |
| Trained-policy benchmark / 训练后策略评估 | Original `scripts/test/eval_unified.py` | Blocked by absent weights / 缺少权重，暂无法执行 |
| Training and physical robot / 训练与实机 | Historical scripts/notes / 历史脚本与笔记 | Not rerun, no hardware or training-time result guarantee / 未重跑，无硬件或训练成绩保证 |

The quick start verifies interfaces and control logic, not the poster's 94% success rate. No core code, YAML geometry, training data or original result values were edited.

快速开始验证接口与控制逻辑，不复现海报的 94% 成功率。未改核心代码、场景几何、训练数据或原始结果数值。

## Environment and setup / 环境与安装

Use Python 3.12 with `requirements.txt`, which points to the fully pinned CPU verification environment in `requirements-lock.txt`. The lock includes NumPy 2.2.6, Matplotlib 3.10.3, IR-SIM (`ir-sim`) 2.10.1, CPU PyTorch and TensorBoard. The import name is `irsim`; the distribution to install is `ir-sim`, correcting the original dependency list. No CUDA is required. Optional historical MARL/Isaac/Habitat/NeuPAN environments are not covered by this lock.

使用 Python 3.12；`requirements.txt` 引用完整固定的 CPU 验证环境 `requirements-lock.txt`。包含 NumPy 2.2.6、Matplotlib 3.10.3、IR-SIM 2.10.1、CPU PyTorch 与 TensorBoard。导入名是 `irsim`，应安装的发行包为 `ir-sim`，已纠正原依赖清单。无需 CUDA；旧 MARL/Isaac/Habitat/NeuPAN 环境不在此锁定范围内。

From repository root / 在仓库根目录：

```sh
python -m venv .venv
# PowerShell
.venv\Scripts\Activate.ps1
# Linux alternative / Linux 替代命令
# source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -X utf8 tools/verify_project.py
```

If activation is blocked, replace `python` with `.venv\Scripts\python.exe` on Windows, or `.venv/bin/python` on Linux. `-X utf8` is required on Windows environments whose default text encoding cannot read the Chinese YAML comments. CI sets `PYTHONUTF8=1` and `MPLBACKEND=Agg`. Rendering windows, a GPU, accounts and credentials are unnecessary. Package downloads need internet; PyTorch is a large download, so allow several GB of free disk and several minutes or longer on a slow connection. Verification uses local resources only, usually seconds to tens of seconds. No billable service or paid GPU is used.

激活受限时直接使用虚拟环境解释器。部分 Windows 默认编码无法读取 YAML 中文注释，需保留 `-X utf8`。CI 设置 UTF-8 与非交互绘图后端。无需图形窗口、GPU、账号或凭据。安装需联网，PyTorch 下载较大，预留数 GB 空间与数分钟或更长时间；验证只用本地资源，通常数秒至数十秒，无付费服务或付费 GPU。

Windows and Linux verification outcomes are recorded in [VALIDATION.md](VALIDATION.md); macOS, CUDA training and robots are not claimed compatible based on these checks.

Windows/Linux 验证结果见记录，不据此声称 macOS、CUDA 训练或实机兼容。

## Expected outputs / 成功判断

`summarize_results.py` prints hard-scenario success rates 67.36% (baseline), 0.00% (archived NeuPAN configuration), 93.75% (STPS v2). It validates three finite rates in [0,1] for each scenario and checks stored means/population standard deviations to their three-decimal precision. Invalid summaries return a nonzero exit status. Exact mean improvement is 26.3889 percentage points; independently rounded poster averages differ by 27 points.

指标脚本输出基线 67.36%、存档 NeuPAN 配置 0.00%、STPS v2 93.75%。检查每场景的三个有限成功率及三位小数精度下的均值/总体标准差，错误数据返回非零状态。精确提升为 26.3889 个百分点；海报分别取整后相差 27。

`smoke_stps.py` fixes seed 42 for Python and NumPy and uses the original v2 definitions without running its top-level weight-loading block. Main policy output `[-1,0]` maps to zero velocities; escape `[-1,1]` rotates in place. In each of U-trap, Double-U, narrow door and corridor, the 20-position stall window triggers escape after 19 main calls; the 60-step check then finishes after 41 escape calls with timeout. `artifacts/smoke_stps.json` explicitly marks `synthetic_policies: true`. The policy stub does not use the LiDAR; the separate model smoke test does.

仿真检查固定种子 42，读取原 v2 定义并跳过顶层权重加载。主策略静止，脱困策略原地旋转；四场景均在 19 次主策略调用后触发切换，60 步检查以 41 次脱困调用和超时结束。JSON 明确标注合成策略。该策略桩不使用雷达，另一个模型检查才验证雷达编码。

`smoke_model.py` seeds PyTorch/NumPy to 42, constructs the original CNNTD3 with `load_model=False`, encodes a real 180-beam scan into 185 values, checks two bounded action outputs and twin scalar Q-values, then executes a simulator step. It also checks infinite scan normalization, terminal flags and rejection of incorrect scan length. Random weights provide no navigation-quality evidence. TensorBoard logs and other generated files stay in `artifacts/`.

模型检查固定种子，初始化原 CNNTD3 而不加载权重，验证真实雷达的 185 维编码、两个有界动作、两个标量 Q 值及仿真步；另检查无限远雷达归一化、终止标志和错误扫描长度。随机权重不提供导航质量证据，生成记录保留本地。

`verify_project.py` runs all of the above plus README local-link and poster checksum checks. Success ends with `VERIFICATION PASSED`; failure exits nonzero.

完整验证器执行上述流程以及 README 本地链接和海报哈希检查，成功打印通过提示，失败返回非零状态。

## Trained evaluation: required missing resources / 训练策略评估缺失资源

```sh
python tools/check_resources.py --strict
```

The checkout lacks all 12 files below under `src/code/robot_nav/models/CNNTD3/checkpoint/`. A missing-resource exit code of 2 is expected; it is not a successful inference run.

该目录缺少下列三组共 12 个权重文件；缺失时返回 2 属预期检查结果，不代表推理成功。

| Prefix / 前缀 | Role / 用途 | Required suffixes / 后缀 |
| --- | --- | --- |
| `CNNTD3` | Baseline / 基线 | `_actor.pth`, `_actor_target.pth`, `_critic.pth`, `_critic_target.pth` |
| `CNNTD3_v7_finetune_best` | Precision / 精确策略 | Same four suffixes / 同四个后缀 |
| `CNNTD3_improved` | Exploration / 探索策略 | Same four suffixes / 同四个后缀 |

There is no published checkpoint download URL or authoritative checksum in this checkout. Obtain the exact team-owned weights with provenance, environment and checksums before running the original benchmark; do not substitute unrelated historical `.pt` models. The history contains older PPO artifacts, which are not these CNN-TD3 weights and were not restored or loaded.

仓库未提供这些模型的下载地址或可信校验值。需由团队提供准确权重、来源、环境与校验值；不能替换为不相关的历史 `.pt`。历史存在旧 PPO 产物，不属于这三组权重，本次未恢复或加载。

Once resources are obtained, the original command is:

```sh
cd src/code
python -X utf8 scripts/test/eval_unified.py
```

This command is currently **blocked**, not a working quick start. It creates models and TensorBoard logs, evaluates both policies on four scenes and writes `unified_comparison.json` into `src/code/`, leaving the archived `src/results/final/` copy intact. The full protocol is 288 episodes (2 methods × 4 scenarios × 3 seeds × 12 starts), at most 500 steps each; runtime is unmeasured here.

该命令当前受阻，不作为可用快速开始。运行后在 `src/code/` 写结果，原成果目录不受覆盖；完整协议为 288 回合，每回合最多 500 步，本次未测运行耗时。

The original evaluator uses seeds `[42,123,2026]` for start perturbations only: x/y ±0.3 m and headings cycling 0, 1.57, 3.14, -1.57 with ±0.4 rad perturbation. It does not seed every noisy LiDAR/global simulator operation. Accordingly, the archived trained results cannot be promised byte-identical even after weights are recovered. V7 also needs absent `CNNTD3_improved_best` starting weights; v6 saves that prefix while the evaluator loads `CNNTD3_improved`. Training provenance/naming must be reconciled by the authors before exact retraining claims.

原脚本只为起点扰动设置三个种子，并未固定雷达噪声及全部仿真随机性，获得模型后也不能承诺结果逐字节一致。v7 另需缺失的 `improved_best` 起始权重；v6 保存名与评估器读取名不同，须由原作者核对训练来源与命名，才能主张严格训练复现。

## Data and reporting caveats / 数据与报告限制

- World YAMLs are the authoritative scene definitions: 10×10, step time 0.3 s, 180-beam noisy LiDAR, 0.3 m goal threshold in the four benchmark scenes. V2 uses an initial escape duration 120 steps, then up to 240 on repeated switches, with a 3× escape-time fallback. / 以 YAML 为准，重复切换及超时以原 v2 代码为准。
- `src/code/robot_nav/assets/data.yml` is the existing 44,054,082-byte offline experience file, versioned at baseline `6b8eee5` with Git blob `1a4eee1fd98bdc9c11b756681ba0c39bb3c965e5`. It is retained unchanged, not required by quick start, and has no separately documented collection/license metadata. Do not infer permission to reuse it from public visibility. / 该离线经验文件以基线提交和 Git 对象标识版本，保留原样，快速开始不读取；缺少独立采集/许可元信息，公开不代表允许复用。
- Archived JSONs store per-seed success rates, not full trajectories, collision counts, timing traces or every raw episode. Standard-environment 88%/87% comes from the final table; those raw 100-episode records are absent. / JSON 不含完整回合与计时轨迹，标准环境成绩仅有表格记录。
- The v3 summary matches v2 on the four scenarios; it does not establish a gain. The standalone sensitivity script uses a 60-step escape duration, unlike unified v2's 120-step initial duration. Poster curves and sensitivity claims lack complete raw logs. / v3 未显示提升；敏感性设置不同，曲线与参数结论缺少完整原始记录。
- NeuPAN's archived 0% is a compact-scene/configuration result; its runner is not included in the unified v2 evaluator. Historical Isaac, Habitat, ROS2 and delay-compensation notes do not establish current reproducibility or hardware safety. / 不将 NeuPAN 配置失败泛化，不将历史笔记当成本次实机证据。
- Original four-element robot states are truncated to three by the tested simulator, and optional keyboard-backend warnings are expected in headless mode. No core YAML was rewritten to silence them. / 状态截断和无键盘后端警告属已观察现象，未为消警告改动核心场景。

All material is subject to [RIGHTS.md](../RIGHTS.md); external dependencies retain their own licenses. / 使用条件见权利说明，依赖遵循各自许可。
