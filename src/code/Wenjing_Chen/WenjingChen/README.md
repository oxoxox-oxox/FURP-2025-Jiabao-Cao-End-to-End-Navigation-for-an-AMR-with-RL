# Downloads Final Snapshot

This folder preserves useful STPS-related files found in `~/Downloads`.

These files are kept as a historical/final-working snapshot because several final experiments were developed or exported there before being copied into the main experiment directory.

## Important Warning

Some downloaded scripts still contain paths such as:

```python
os.chdir('/root/DRL-robot-navigation-IR-SIM')
sys.path.insert(0, '/root/DRL-robot-navigation-IR-SIM/robot_nav')
```

On the current machine, the correct path is usually:

```text
~/DRL-robot-navigation-IR-SIM
```

For runnable versions, prefer the cleaned scripts in:

```text
code/evaluation/
code/figure_generation/
```

Use this folder when you need to trace what was in Downloads or recover an exact older version.

## Folder Meaning

```text
stps_evaluation/
  eval_stps.py
  eval_stps_v2.py
  eval_stps_sensitivity.py
  eval_v7.py
  eval_unified.py
  eval_expanded.py
  eval_diagnose.py
  eval_fair_comparison.py
  eval_neupan.py
  script1_full_comparison.py
  script3_stps_v3.py

training_ablation/
  rl_train_v6_anneal.py
  rl_train_v7_finetune.py
  rl_train_v2_continue_v2.py
  rl_train_v5_combined_v2.py
  script2_train_utrap.py

scenario_tests/
  test_u_trap_cnntd3.py
  test_rcpg_hard_scenarios.py
  final_evaluate_generalization.py

neupan_delay/
  NeuPAN observation/inference delay tests and compensation experiments.

configs/
  World, planner, robot, and DUNE training yaml files used in NeuPAN/STPS experiments.

real_robot_ros/
  ROS/NeuPAN node variants for real-robot or deployment-oriented experiments.

model_files/
  CNNTD3.py snapshot from Downloads.
```

## Poster Mapping

| Poster claim / section | Downloads snapshot files |
|---|---|
| STPS v2 final method | `stps_evaluation/eval_stps_v2.py` |
| STPS parameter sensitivity | `stps_evaluation/eval_stps_sensitivity.py` |
| Precision policy / annealed policy | `stps_evaluation/eval_v7.py`, `training_ablation/rl_train_v6_anneal.py`, `training_ablation/rl_train_v7_finetune.py` |
| Unified comparison | `stps_evaluation/eval_unified.py`, `stps_evaluation/script1_full_comparison.py` |
| STPS v3 did not improve | `stps_evaluation/script3_stps_v3.py` |
| U-trap diagnosis | `stps_evaluation/eval_diagnose.py`, `scenario_tests/test_u_trap_cnntd3.py` |
| NeuPAN / delay exploratory branch | `neupan_delay/`, `real_robot_ros/`, `configs/` |

## Final Project Recommendation

For the final poster/report:

- Present **STPS v2** as the final method.
- Use **eval_unified / unified_comparison.json** for the main table.
- Use **evidence.py / full_rl_metrics.json** for supplementary RL metrics.
- Treat NeuPAN delay and real-robot scripts as exploratory, not as the main contribution.

