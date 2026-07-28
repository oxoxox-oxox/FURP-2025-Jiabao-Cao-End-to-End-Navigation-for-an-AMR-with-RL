# Weekly Progress Log

> Update this file **every week**. Add a new entry at the top for each week.
> This is the first thing we check during review. Keep it honest and specific — it also feeds your attendance record (Rule 1).

**How to use:** copy the *Week template* block below for each new week. Newest week goes at the top.

---

## Week template — copy me

### Week N — YYYY-MM-DD

**Attended this week's meeting:** Yes / No (if No, did you email leave? Yes / No)

**Progress this week**
- _What did you actually do / finish?_

**Challenges & blockers**
- _What got in the way? What are you stuck on?_

**Next steps**
- _What will you do next week?_

**Hours spent (optional):** _e.g. 6h_

**Links (optional):** _commits, notebooks, docs, datasets..._

---

<!-- =================  YOUR ENTRIES BELOW  ================= -->

### Week 7 — 2026-07-20

**Attended this week's meeting:** Yes

**Progress this week**

- Identified a core conflict in CNNTD3 navigation: policies trained for exploration can escape U-shaped traps, while policies trained for precision can pass narrow doors, but a single policy struggled to keep both abilities.
- Ran annealing experiments by loading the exploration policy and linearly reducing exploration reward from 0.5 to 0 across 30 epochs.
  - Narrow-door success recovered to 100%.
  - U-trap success dropped back to 0%.
  - Conclusion: the exploration behavior was not retained under precision-oriented fine-tuning.
- Proposed **STPS: Stall-Triggered Policy Switching** as a runtime method instead of further trying to force one policy to solve all cases.
  - Stall detection: displacement below 0.15 m over 20 steps.
  - Oscillation detection: at least 5 direction reversals in 12 steps.
  - Escape mode lasts 120 steps and switches back after displacement exceeds 0.5 m.
- Performed parameter sensitivity tests over a 3 x 3 grid of window size and displacement threshold.
  - Narrow-door success remained 100% across all tested settings.
  - Window size 20 gave stable U-trap performance around 67%.
- Final comparison used 3 seeds, 12 perturbed starts per scenario, and a 100-episode standard-environment evaluation.

| Method | Standard | U-trap | Double-U | Narrow Door | Corridor | Scenario Avg. |
|---|---:|---:|---:|---:|---:|---:|
| CNNTD3 baseline | 87% | 0±0% | 69±4% | 100±0% | 100±0% | 67% |
| NeuPAN | 0% | 0±0% | 0±0% | 0±0% | 0±0% | 0% |
| **STPS v2** | **88%** | **75±7%** | **100±0%** | **100±0%** | **100±0%** | **94%** |

- Confirmed that the previously reported CNNTD3 baseline SR=92% came from training-time evaluation with only 10 episodes per epoch; the independent 100-episode evaluation gives SR=87%.
- Verified that NeuPAN failed in the compact 10 x 10 m evaluation scenarios. The likely cause is that MPC outputs negative linear velocity, which is clipped by the forward-only constraint, and `d_max=1.0m` is too conservative for compact scenes.
- Tried two additional variants that did not improve results:
  - U-trap specialist model: best SR=83%, no improvement over the existing improved policy.
  - STPS v3 with goal-distance check, gradual escape, and cooldown: 75±7%, same as v2 but with more complexity.
- Ran two PINN demos: a basic ODE example and a differential-drive kinematics example. Current conclusion: PINN is less useful than hand-written integration in the clean simulator, but may matter for real-robot deployment.
- DC-NeuPAN direction: tested head-on dynamic obstacle scenario over 30 seeds. Kalman prediction was harmful in head-on cases, reducing SR from 63% to 50%, because the constant-velocity assumption fails when obstacle behavior changes abruptly.

**Challenges & blockers**

- NeuPAN produced 0% SR in the 10 x 10 m benchmark. Re-training a 0.4 x 0.4 robot model still did not solve the issue.
- The 4070 workstation was not fully configured due to missing monitor, keyboard, and network connection.
- STPS v3 and specialist training did not improve over STPS v2.
- The delay-compensation direction likely needs real-robot tests to become convincing.

**Next steps**

1. Stop adding new research branches and finalize the STPS story as the main contribution.
2. Package the final comparison table, sensitivity table, and 3 successful / 3 failed trajectory cases.
3. Keep NeuPAN and DC-NeuPAN as secondary negative results or discussion, not the main evaluation.
4. Prepare final poster/report and a 5-8 minute demonstration video.

**Hours spent:** Not fully recorded.

**Links:**
- STPS: `eval_stps_v2.py`, `eval_expanded.py`, `eval_unified.py`, `eval_diagnose.py`
- Parameter sensitivity: `stps_sensitivity_results.json`
- Unified comparison: `unified_comparison.json`
- PINN demos: `pinn_01_basic_ode.py`, `pinn_02_diff_drive_kinematics.py`
- DC-NeuPAN head-on test: `test_neupan_delay_eval_v4.py`

---

### Week 6 — 2026-07-13 / 2026-07-15

**Attended this week's meeting:** Yes

**Progress this week**

- Tested whether NeuPAN observation-delay robustness could be solved through training.
- Used a monkey patch to modify `DUNETrain.generate_data_set`, injecting delayed point coordinates into the training data while keeping the true distance labels unchanged.
- Trained two DUNE variants:
  - fixed 500 ms delay training;
  - random 0-500 ms delay training.
- Compared the original, fixed-delay, and random-delay models across four evaluation scripts.

| Version | Random Delay without Compensation | Offset after Compensation |
|---|---:|---:|
| Original | 1.63 | 0.033 |
| Fixed 500 ms training | 1.49 | 0.036 |
| Random delay training | 1.83 | 0.039 |

- Conclusion: training-side DUNE modification did not solve the delay problem.
- Tested robot-state kinematic prediction and obstacle point-cloud extrapolation.
  - Robot-state prediction worked well in static scenes.
  - Obstacle point-cloud linear extrapolation was weak in both static and dynamic scenes.
- Replaced raw point-cloud extrapolation with a Kalman tracker.
- Fixed three implementation bugs in the delay-compensation experiments:
  - Tracker update bug: the tracker was only updated when `d_steps == 0`, which never happened under delay. Fixed by updating from delayed point cloud every step and predicting forward when needed.
  - Static point drift bug: wall points were assigned non-zero velocity. Added velocity and cluster filtering.
  - Coordinate-frame mismatch: point cloud was decoded using delayed state while planning state used compensated state. Fixed by decoding point cloud in the compensated frame.

Static-scene results:

| Condition | SR% | Straight Segment Offset |
|---|---:|---:|
| No-delay baseline | 100 | 0.0005 |
| Random 100-1000 ms delay, no compensation | 100 | 1.4929 |
| Random 100-1000 ms delay + robot-state compensation | 100 | 0.0004 |
| Random 100-1000 ms delay + additional tracker branch | 100 | 0.7821 |

Dynamic-scene results on deterministic crossing scenario, n=10:

| Condition | SR% | Collision% |
|---|---:|---:|
| No-delay baseline | 100 | 0 |
| Delay, no compensation | 10-20 | 80-90 |
| Delay + robot-state compensation | 100 | 0 |
| Delay + robot-state compensation + Kalman | 100 | 0 |

- Key conclusion: robot-state compensation alone almost restored baseline performance in the deterministic crossing scene. Kalman tracking did not add clear benefit in this setting.
- Began real-robot deployment by adding delay-compensation logic to the official `neupan_ros2` node.

**Challenges & blockers**

- Dynamic tests were inconsistent until seeds were controlled using `random`, `numpy`, and `torch`, and thread nondeterminism was reduced with `OMP_NUM_THREADS=1`.
- Real-robot NeuPAN behavior was not usable yet:
  - Planner used LIMO speed parameters on a TurtleBot3 Burger, causing severe model-execution mismatch and oscillatory motion.
  - DUNE model size and safety margin were too large for Burger, leading to early stopping in front of obstacles.

**Next steps**

1. Run real-robot A/B tests with compensation off/on.
2. Record rosbag and odometry trajectories for real delay analysis.
3. Evaluate the head-on dynamic obstacle scenario over more seeds.
4. Add fixed-delay scans at 0/200/400/600/800/1000 ms if time allows.

**Hours spent:** Not fully recorded.

**Links:**
- Test scripts: `test_neupan_delay_eval_v3.py`, `test_neupan_delay_eval_v4.py`
- Scenario files: `env_turn_simple.yaml`, `env_turn_fast_dynamic.yaml`, `env_headon.yaml`
- DUNE training: `dune_train_delay.py`
- Real-robot node: `dc_neupan_node.py`, `robot.yaml`, `planner.yaml`
- Data archive: `~/paper_data/{raw_json,frozen_v1}`

---

### Week 5 — 2026-07-06 / 2026-07-07

**Attended this week's meeting:** No meeting held.

**Progress this week**

- Explored the delay-aware navigation direction and started NeuPAN delay-injection experiments.
- Set up Isaac Sim 6.0.1 and Isaac Lab.
- Ran an initial NeuPAN action-delay experiment by adding an action buffer between planner output and execution.
  - Later concluded this was not the real delay setting, because real delay usually occurs between observation and agent/planner.
- Separated inference delay from observation delay and tested action chunking.
  - Inference delay: the planner runs less often and repeats old actions or executes a pre-planned action chunk.
  - Observation delay: the planner still runs every step, but receives stale observations.
- Key finding: action chunking helps inference delay but does not solve observation delay, because the planner input itself is outdated.

Inference-delay results:

| Condition | SR% | Oscillation | Lateral Offset |
|---|---:|---:|---:|
| No-delay baseline | 100 | 66 | 0.0 |
| 5-step inference delay, no chunking | 100 | 26 | 0.14 |
| No delay + chunking | 100 | 10 | 0 |
| 5-step inference delay + chunking | 100 | 6 | 0.0 |

Observation-delay results:

| Condition | SR% | Oscillation | Lateral Offset |
|---|---:|---:|---:|
| No-delay baseline | 100 | 66 | 0.0 |
| 5-step observation delay, no chunking | 100 | 17 | 1.81 |
| No delay + chunking | 100 | 6 | 0 |
| 5-step observation delay + chunking | 100 | 10 | 1.91 |

- Implemented robot-state kinematic compensation for observation delay.

| Condition | SR% | Steps | Lateral Offset |
|---|---:|---:|---:|
| 5-step / 500 ms delay, no compensation | 100 | 229 | 1.81 |
| 5-step delay + state prediction | 100 | 149 | 0.0 |
| 10-step / 1000 ms delay, no compensation | 100 | 853 | 4.03 |
| 10-step delay + state prediction | 100 | 150 | 0.0 |

- Identified that the perfect compensation result was too ideal because delay steps were fixed, the simulator kinematics matched exactly, and static LiDAR did not change.
- Converted delay from fixed steps to millisecond-level random delay sampled each step.

| Condition | SR% | Steps | Lateral Offset |
|---|---:|---:|---:|
| No-delay baseline | 100 | 0 | 0 |
| Random 100-1000 ms delay, no compensation | 100 | 179 | 1.94 |
| Random 100-1000 ms delay + compensation | 100 | 150 | 0.04 |

- Added a dynamic-obstacle scene and tested L1 compensation: robot-state prediction plus obstacle point-cloud linear extrapolation.

| Method | Collision Rate | Steps | Lateral Offset |
|---|---:|---:|---:|
| No-delay baseline | 100% | 172 | 0.37 |
| Delay, no compensation | 100% | 246 | 1.86 |
| Robot-state compensation only | 100% | 210 | 0.30 |
| L1 full compensation | 20% | 832 | 0.11 |

- Key conclusion: L1 compensation reduced collision rate but caused over-conservative behavior and timeout, suggesting that simple constant-velocity extrapolation is not sufficient for dynamic obstacles.

**Challenges & blockers**

- Initial dynamic obstacle scenarios were too hard: even no-delay NeuPAN achieved SR=0%.
- Some 0-delay dynamic results differed across groups because obstacle initialization was random.
- Fairness issue remains: it is unclear whether NeuPAN should be re-trained, compensated externally, or compared as-is.
- Linear obstacle extrapolation is not robust to turning or accelerating obstacles.

**Next steps**

1. Test whether NeuPAN training can solve delay robustness.
2. Add dynamic obstacle tests to expose the limits of state prediction.
3. Add sensor noise and model uncertainty if time allows.
4. Decide whether the final project should focus on RL policy improvement or NeuPAN delay compensation.

**Hours spent:** Not fully recorded.

**Links:**
- Delay tests: `test_neupan_delay_vis.py`, `test_neupan_delay_fin.py`, `test_neupan_unified.py`, `test_neupan_obs_delay.py`
- Compensation tests: `test_neupan_compensate.py`, `test_neupan_ms_delay.py`, `test_neupan_L1.py`, `test_neupan_dynamic.py`

---

### Week 4 — 2026-06-29 / 2026-06-30

**Attended this week's meeting:** Yes

**Progress this week**

- Trained additional CNNTD3 variants to test whether curriculum learning and exploration reward could improve hard-scenario generalization.
- Trained CNNTD3_v3 with conservative curriculum for 100 epochs. Standard SR was around 70-80%, less stable than the improved variant.
- Trained ATD3 with multi-head self-attention replacing the CNN encoder. This failed to converge and was abandoned.
- Trained CNNTD3_v4_improved with distance-shaped reward, 120 epochs, and best-checkpoint saving. Final SR was around 75%, with no clear improvement over baseline.
- Re-trained NeuPAN DUNE for TurtleBot3 Burger size, but the retrained model still could not pass the corridor-navigation case.
- Ran systematic final evaluation across 20 generalization trials and four hard scenarios: U-trap, Double-U, narrow door, and corridor.
- Found that CNNTD3_v2 gave the best generalization SR around 90%, suggesting that exploration reward strength 0.15 may be a better trade-off than 0.3 or 0.1.
- Tried continuing CNNTD3_v2 training for 40 more epochs, but it did not improve.
- Designed CNNTD3_v5_combined to combine moderate exploration reward and earlier curriculum. It helped U-trap cases but weakened generalization.
- Read additional papers to improve research positioning, including NeuPAN and recent navigation / VLA references.

**Challenges & blockers**

- ATD3 failed to converge, possibly because the world was too large and the goal often exceeded LiDAR range.
- Long evaluation runs hit X11 display-connection limits after creating hundreds of simulator instances; fixed by reusing simulator instances inside each test phase.
- Accidentally deleted `robot_nav/runs/`, so some TensorBoard curves were lost, although model checkpoints remained.
- IR-SIM crashed during v5 training due to a spatial-index mismatch after scene objects changed during runtime.
- Direction was still scattered across multiple possible stories.

**Next steps**

1. Continue small-scale model tuning only if it directly supports the final story.
2. Read more papers to position the contribution.
3. Prefer an analysis-and-system-comparison story over another large training run.

**Hours spent:** Not fully recorded.

**Links:**
- Final evaluation figure: `../src/evaluate.png`
- NeuPAN paper: https://arxiv.org/abs/2403.06828

---

### Week 3 — 2026-06-22

**Attended this week's meeting:** Yes

**Progress this week**

- Trained `CNNTD3_improved` with curriculum learning and exploration reward for 60 epochs.
- Trained a curriculum-only ablation for 60 epochs.
- Evaluated CNNTD3, RCPG, curriculum-only, and curriculum + exploration across structured hard scenarios.
- Completed an ablation study separating curriculum learning from exploration reward.
- Built a TensorBoard comparison across trained models.

| Scenario | CNNTD3 | RCPG | Curriculum Only | Curriculum + Exploration |
|---|---:|---:|---:|---:|
| Standard env | 92% | 88% | ~81% | ~78% |
| U-trap | 0% | 0% | 0% | 100% |
| Double-U | 33% | 0% | 67% | 33% |
| Narrow door | 4.8% | 90.5% | 9.5% | 0% |
| Symmetric corridor | 83% | 100% | 100% | 100% |

- Key findings:
  - Exploration reward was the critical factor for escaping the U-trap.
  - Curriculum learning alone improved Double-U and corridor cases.
  - Hard-scenario improvement reduced standard-environment SR.
  - No single model dominated all scenarios.
- Analyzed why standard SR dropped after hard-scenario training:
  - training budget dilution;
  - exploration reward side effects;
  - reward-distribution shift.

**Challenges & blockers**

- Computer shut down during overnight training, losing partial progress.
- Exploration reward parameters were manually selected and need sensitivity testing.
- The S4 dead-end maze was too restrictive and was dropped from final evaluation.
- Debugging and experiment tracking were not systematic enough.

**Next steps**

1. Read related work on local-minima escape.
2. Design generalization tests with unseen U-trap variants.
3. Increase training budget or introduce better experience balancing if continuing training.

**Hours spent:** Not fully recorded.

**Links:**
- Training logs: `cnntd3_improved_train.log`, `curriculum_only_train.log`
- Test results: `improved_hard_scenario_results.csv`
- Test scripts: `test_improved_hard_scenarios.py`, `test_curriculum_only.py`

---

### Week 2 — 2026-06-15

**Attended this week's meeting:** Yes

**Progress this week**

- Completed Habitat PPO PointNav baseline with approximate SR=0.85 and SPL=0.65.
- Ran two reward-shaping experiments; both underperformed the baseline.
- Tested NeuPAN on three scenarios as a model-based comparison.
- Built five structured hard-scenario environments in IR-SIM.
- Evaluated CNNTD3 and RCPG on the hard scenarios.
- Found that GRU memory is a double-edged sword: it improves narrow-door precision and symmetry breaking, but hurts concave-trap recovery.

**Challenges & blockers**

- PyTorch 2.6 checkpoint incompatibility was fixed with `weights_only=False`.
- NeuPAN dependency conflicts required a separate conda environment.
- IR-SIM wall placement and TD3 class/state-dimension mismatch took significant debugging time.
- RCPG training was much slower than CNNTD3.

**Next steps**

1. Implement curriculum learning with U-trap scenarios.
2. Add exploration reward to penalize revisiting the same area.
3. Evaluate the improved model across all hard scenarios.

**Hours spent:** Not fully recorded.

**Links:**
- Training curve: `../src/training_curve.png`
- Comparison curve: `../src/comparison_curve.png`
- Hard-scenario scripts and CSV results in `../src/`

---

### Week 1 — 2026-06-10

**Attended this week's meeting:** Yes

**Progress this week**

- Set up repository from the FURP template.
- Read the paper of NeuPAN
- Try my hand on habitat lab

**Challenges & blockers**

- The methametical formuler in NeuPAN paper is too hard to understand
- Limited space for my Ubuntu system.

**Next steps**

- Try to learn more methametical formular in paper.
- Buy a SSD to install a new Ubuntu 22.04 to use.

**Hours spent (optional):**

**Links (optional):**
