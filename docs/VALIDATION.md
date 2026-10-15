# Validation record / 验证记录

Date / 日期: 2026-10-15 (Asia/Shanghai). Baseline / 原始版本: `6b8eee5`.

## Baseline inspection / 修改前检查

- Original checkout was clean; a full private copy including `.git` and all checked-out files was created before edits. / 修改前工作区干净，已完整备份含 Git 历史的检出目录。
- Archived per-seed arithmetic yields baseline 67.3611%, STPS v2 93.75%, improvement 26.3889 percentage points. Stored means and population stds agree within their rounding precision. / 存档核算与原汇总精度一致。
- The original `SIM` wrapper reset and stepped all four benchmark worlds using IR-SIM 2.10.1, exposing 180 LiDAR ranges each. Windows default text decoding failed on Chinese comments; `-X utf8` resolved this without editing any YAML/code. / 四场景原接口可运行；UTF-8 启动解决中文编码问题，无需修改核心文件。
- No current CNN-TD3 `.pth` weights are included. Older `.pt` models in history are not substitutes and were not restored or loaded. / 当前无所需策略权重，旧历史模型不替代、不加载。

## Verification scope / 验证范围

Seven regression tests exercise original v2 oscillation, seeded starts, escape return, no-progress timeout, goal/collision termination and result validation. Real-world simulation checks use synthetic policies; the model interface smoke check uses random initial weights. Neither is trained-policy performance reproduction. Original source/data/results remain protected separately from documentation/helper changes.

7 项回归测试覆盖原 v2 的振荡、起点种子、脱困返回、超时、成功/碰撞终止和结果核算。仿真使用合成策略，模型接口使用随机权重，均不作为训练策略成绩复现。原始源码、数据和结果与文档/辅助工具分别保护。

Execution and publication outcomes are appended only after they are observed. / 实际运行与发布结果在确认后追加。

## Executed locally / 本地已执行

- New isolated environment on Windows 11 Home 10.0.26200, Python 3.12.14, CPU PyTorch 2.8.0+cpu, IR-SIM 2.10.1, NumPy 2.2.6, Matplotlib 3.10.3, TensorBoard 2.20.0. Exact direct/transitive versions are in `requirements-lock.txt`. `pip check` passed. / 新建隔离环境及依赖检查通过，完整固定版本见锁文件。
- `python -X utf8 tools/verify_project.py` passed all seven regression tests, archived aggregate validation, README local links (including poster HTML links), poster hashes, four real IR-SIM scenes and the original CNN-TD3 actor/twin-critic interface. / 完整验证覆盖测试、指标、链接、哈希、四场景及原模型接口，全部通过。
- In each real scene, stationary main/rotating escape synthetic policies produced exactly 19 main calls and 41 escape calls over 60 steps, terminating in timeout. This checks switching, not learned escape ability. / 合成策略切换调用数与预期一致，不证明学习后的脱困能力。
- Random-weight CNN-TD3 accepted 180 LiDAR ranges, encoded 185 values, returned two finite bounded actions and two finite scalar Q-values, and executed a simulation step; normalization/terminal/dimension checks passed. / 随机权重的编码、输出、步进及异常维度检查通过。
- `tools/check_resources.py --strict` returned expected exit 2 and listed 12 absent files. No trained-policy performance, training run, hardware execution or timing claim was verified. / 资源检查准确报告缺少 12 个文件，不把受阻流程写成成功。
- All 60 protected source, scene, data and original result files were byte-identical to the pre-edit private backup; Git reports no corresponding core changes against `6b8eee5`. The ~44 MB offline dataset remains tracked. / 核心文件字节一致，大数据文件保留。
- Scanned 744 original Git history objects for common private-key/token patterns, with no pattern hits. This bounded scan does not prove absence of every possible secret format. Credentials were not added. / 常见凭据模式扫描无命中，但不宣称覆盖所有秘密格式，未新增凭据。

Logs are excluded under `artifacts/` or `*.log`. Expected nonfatal simulator warnings concern four-to-three state truncation and unavailable optional keyboard input. The model check also emits a PyTorch tensor-construction performance warning; it does not affect assertions or core code. / 日志排除上传，已观察到的非致命警告不影响验证结论。
