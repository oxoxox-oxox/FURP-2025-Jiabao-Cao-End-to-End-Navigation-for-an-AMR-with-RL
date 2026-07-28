# Final Submission Checklist

This checklist maps the project requirements to the current evidence.

## Required Items

| Requirement | Current Status | Evidence | Final-week Action |
|---|---|---|---|
| Runnable baseline command | Partial | `src/code/Wenjing_Chen/CNNTD3/README.md` has older train/test commands; `final_stps/` has final scripts | Add exact final commands to README/poster/report |
| Documented environment | Partial | Weekly logs mention Ubuntu 22.04, RTX GPU, ROS 2, Habitat, IR-SIM, NeuPAN env separation | Write a short environment section with Python/PyTorch/IR-SIM/GPU |
| Validation metric table | Yes | `src/results/final/final_results_table.md`, `unified_comparison.json` | Use this as the main result table |
| 3 success + 3 failure cases | Written template, images still needed | `docs/case_analysis_template.md` | Export or screenshot 6 representative trajectories |
| Focused improvement or ablation | Yes | STPS v2; earlier curriculum/exploration ablation; STPS v3 negative result | Present STPS v2 as final method; include one ablation table |
| Short explanation of what did not work | Yes | Week logs, `final_week_recovery_plan.md`, `case_analysis_template.md` | Keep it short: reward shaping, single-policy training, STPS v3, NeuPAN compact mismatch |

## RL Navigation Metrics

| Metric | Current Status | Evidence / Note | Need To Supplement? |
|---|---|---|---|
| Success rate | Yes | Final table and JSON results | No |
| Episode return | Partial | Older CSVs contain `reward`; final STPS table does not | Optional: add average reward if easy |
| Collision rate | Partial | Older CSVs contain `outcome=collision`; final table reports SR only | Recommended: report collision/timeout counts for final 6 cases or final evaluation |
| Average path length | Partial | CSVs contain `steps`; path length in meters not consistently logged | Use average steps as proxy, or add path length if script can log positions |
| Path efficiency | Missing / optional | No consistent shortest-path ratio found | If no shortest path is available, report "not available; avg steps used as efficiency proxy" |
| Training curve across seeds | Partial | Training/progress images exist; final seed-level evaluation exists | Do not retrain; show one training curve + 3-seed evaluation bars |
| Qualitative rollouts | Needs final images | Existing few GIFs/images; final STPS cases need screenshots | Yes, export 6 case images/GIFs |

## Minimum Additions Before Submission

1. **Add one command block**:

```bash
cd ~/DRL-robot-navigation-IR-SIM
python eval_unified.py
python eval_stps_v2.py
```

If the actual commands need a conda environment, write:

```bash
conda activate <env_name>
cd ~/DRL-robot-navigation-IR-SIM
python eval_unified.py
python eval_stps_v2.py
```

2. **Add one environment block**:

```text
OS: Ubuntu 22.04
Simulator: IR-SIM
Main model: CNNTD3 / TD3-based continuous-control navigation policy
Python/PyTorch: record from local environment
GPU: record from local machine
Random seeds: 3 seeds for hard-scenario comparison
Evaluation: 12 perturbed starts per scenario; 100 episodes for standard environment
```

3. **Export six visuals**:

- STPS success in U-trap.
- STPS success in Double-U.
- STPS success in narrow door.
- CNNTD3 failure in U-trap.
- exploration-policy failure in narrow door.
- NeuPAN compact-scene failure or table screenshot.

4. **Add one compact failure table**:

| Attempt | Result | Lesson |
|---|---|---|
| Collision reward shaping | hurt learning / made policy cautious | naive reward shaping can damage exploration |
| Single exploration policy | improves U-trap, hurts narrow door | exploration and precision conflict |
| STPS v3 | no gain over v2 | added complexity was not useful |
| NeuPAN compact benchmark | 0% in tested setup | likely domain/configuration mismatch |

## What Is Enough

For this project, it is acceptable if path efficiency is not a full SPL-style metric, as long as the report is honest:

> We report success rate as the primary metric. Collision/timeout outcome and average steps are used as secondary navigation-efficiency indicators. Exact shortest-path efficiency was not available in the current IR-SIM evaluation pipeline.

