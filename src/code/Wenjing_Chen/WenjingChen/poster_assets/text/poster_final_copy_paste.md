# Final Poster Copy-Paste Text

This file is the minimal final version to paste into the academic poster.

## What Data Still Needs Supplementing

### Must Add / Confirm

1. **Author and affiliation**
   - Replace `[Author Name]`, `[University Name]`, `[email]`.

2. **Experiment setup**
   - Simulator, robot type, sensor, action space, scenarios, seeds, evaluation rule.

3. **RL metrics note**
   - Main poster can use Success Rate as the primary metric.
   - Add one sentence explaining that collision/timeout and steps are used as diagnostics.

4. **3 success + 3 failure cases**
   - Already generated in `docs/img/cases/`.
   - Put all six images in the case section.

5. **Future Work**
   - Current poster must not leave this blank.

### Nice To Add

1. Full RL metrics table:
   - SR
   - collision rate
   - average reward
   - average path length
   - path efficiency
   - average steps

2. References footer:
   - 4-6 papers only.

3. Limitations:
   - Short and honest.

## Top Header

### Title

Stall-Triggered Policy Switching for DRL Navigation in Trap Scenarios

### Subtitle

Runtime composition of complementary DRL policies for reducing catastrophic failures in trap-like navigation scenarios

### Author Line

Wenjing Chen · University of Nottingham Ningbo China · Summer Research 2026

### Metric Cards

94%  
Scenario avg SR  
(+27 pp vs baseline)

88%  
Standard env SR  
(matches baseline)

0  
Hard-scenario 0%-SR cells

0  
Additional training cost

## 1. Abstract

Deep reinforcement learning (DRL) enables mapless mobile robot navigation from local sensory observations, but reactive policies can still fail catastrophically in trap-like layouts where reaching the goal requires temporarily moving away from it. In our IR-SIM benchmark, a CNNTD3 baseline achieves 87% success in the standard environment but 0% in the U-trap scenario. We identify an exploration-precision conflict: exploration reward improves U-trap escape but destroys narrow-door precision, while annealing restores precision but loses escape behavior. To address this, we propose Stall-Triggered Policy Switching (STPS), a training-free runtime mechanism that switches between a precision policy and an exploration policy only when stagnation or oscillation is detected. Across four hard scenarios and three seeds, STPS improves scenario-average success from 67% to 94% while preserving standard-environment performance at 88%.

## 2. Introduction

Mapless DRL navigation maps local observations, such as LiDAR and relative goal state, directly to continuous robot actions. This is attractive for AMRs because it avoids explicit map building and can react online in unknown environments. However, purely reactive policies often lack long-horizon recovery behavior. In concave traps, the goal direction may point through an obstacle, so greedy goal tracking causes freezing, oscillation, or timeout. This project asks whether a lightweight runtime mechanism can recover from such local minima without sacrificing precision in narrow passages.

## 3. Key Finding

Exploration reward fixes U-trap escape but destroys narrow-door precision. Annealing the exploration reward recovers narrow-door performance but loses U-trap escape. This shows that the main issue is not only model capacity, but a conflict between two behavioral modes: exploration for recovery and precision for alignment.

## 4. Method: STPS

STPS uses a precision policy by default. When the robot shows evidence of stagnation or oscillation, it temporarily switches to an exploration policy. After sufficient escape progress, it switches back to the precision policy.

Algorithm:

1. Run precision policy by default.
2. If 20-step displacement is below 0.15 m, or direction reversal occurs at least 5 times in 12 steps, switch to escape mode.
3. Execute the exploration policy for at least 120 steps.
4. Switch back when displacement from the switching point exceeds 0.5 m.

Replace the three small method bullets with:

- No extra training: both policies are frozen.
- Low overhead: only position-history checks are added.
- Deployment-friendly: simple runtime supervisor above the learned controller.

## 5. Experiment Setup

Environment: IR-SIM 2D indoor navigation simulator with 10 m x 10 m worlds.  
Robot: differential-drive AMR with continuous linear and angular velocity control.  
Sensor: 180-beam 2D LiDAR plus relative goal information.  
Scenarios: one standard random-obstacle environment and four hard scenarios: U-trap, Double-U, narrow door, and symmetric corridor.  
Baselines: CNNTD3 baseline, RCPG (GRU), curriculum-only policy, exploration policy, annealed precision policy, and NeuPAN configuration test.  
Evaluation: hard scenarios use 3 seeds x 12 perturbed starts per scenario; the standard environment uses 100 episodes. Success Rate (SR) is the primary metric. Collision/timeout outcome and average steps are used as diagnostic metrics.

## 6. Results Caption

STPS is the only tested method that removes all 0%-SR hard-scenario cells while preserving standard-environment success. The largest gain appears in concave-trap scenarios, where switching introduces temporary retreat behavior absent from the precision-only baseline.

## 7. Case Analysis Caption

The failure cases show three different limitations of single-policy navigation: the baseline cannot retreat in the U-trap, the exploration policy loses narrow-door precision, and the annealed policy recovers precision but loses U-trap escape. The success cases show that STPS activates exploration only when recovery is needed and then returns to precision navigation.

Short labels for six images:

- Failure: baseline cannot retreat in U-trap.
- Failure: exploration policy collides in narrow door.
- Failure: annealed policy loses U-trap escape.
- Success: STPS switches to escape mode in U-trap.
- Success: STPS preserves narrow-door precision.
- Success: STPS keeps standard navigation performance.

## 8. Parameter Sensitivity Caption

Narrow-door SR remains 100% across tested thresholds, indicating that switching does not harm precision. U-trap performance is most stable around a 20-step stall window, suggesting that too-short windows trigger prematurely while too-long windows delay recovery.

## 9. Error Taxonomy

Greedy tracking: the precision policy follows the goal direction and cannot retreat from concave traps.  
Reward conflict: exploration improves escape but damages precise alignment.  
Parameter-space limit: a single annealed policy cannot keep both behaviors.  
Detection latency: STPS may miss some U-trap cases because stall confirmation requires 20 steps.

What did not work:

- STPS v3 with goal distance, progressive escape, and cooldown did not improve over v2.
- U-trap specialist training reached similar performance to the exploration policy but did not solve the precision conflict.
- NeuPAN results in compact 10 m x 10 m scenes should be interpreted as a configuration/domain mismatch, not a general failure of NeuPAN.

## 10. Application

STPS can be used as a lightweight supervisory layer for AMR local navigation in warehouses, offices, corridors, and service-robot environments. It is especially useful when robots encounter cul-de-sacs, narrow passages, or temporary local minima. Because STPS uses frozen policies and simple position-history checks, it can wrap an existing learned controller without retraining. This makes it suitable for sim-to-real deployment studies where the switching logic can use odometry or localization history.

## 11. Conclusion

1. We reproduced and diagnosed a CNNTD3 navigation baseline that performs well in standard scenes but fails in trap-like layouts.
2. We identified an exploration-precision conflict: exploration helps local-minimum escape but damages narrow-door alignment.
3. STPS resolves this conflict by composing two frozen policies at runtime using stall and oscillation evidence.
4. In our hard-scenario benchmark, STPS improves scenario-average SR from 67% to 94% and removes all 0%-SR hard-scenario cells while preserving standard-environment performance.

## 12. Future Work

1. Validate STPS on a real TurtleBot/LIMO platform with odometry-based stall detection.
2. Replace hand-designed thresholds with a learned switching trigger based on trajectory history.
3. Extend STPS to dynamic obstacles and delay-aware navigation.
4. Evaluate on randomized trap geometries and unseen narrow-passage layouts.
5. Integrate STPS with ROS 2 Nav2 as a recovery behavior or local-controller supervisor.

## 13. Limitations

STPS currently uses hand-designed switching thresholds and is mainly evaluated in 2D simulation. Exact shortest-path efficiency is not available for all runs, so average steps are used as a navigation-efficiency proxy. Real-robot robustness and dynamic-obstacle performance remain future work.

## 14. References

Tai et al., Virtual-to-real deep reinforcement learning for mapless navigation, 2017.  
Fujimoto et al., Addressing Function Approximation Error in Actor-Critic Methods, 2018.  
Liang et al., Context-Aware Deep Reinforcement Learning for Autonomous Robotic Navigation, 2023.  
Hu et al., DRL-based mapless navigation with local optima, 2024.  
Han et al., NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning, 2025.  
Khatib, Real-time obstacle avoidance for manipulators and mobile robots, 1986.

## Recommended Layout

Use this order from top to bottom:

1. Title + metric cards.
2. Left top: Abstract + Introduction.
3. Right top: Key finding table.
4. Middle: STPS method flowchart.
5. Before Results: compact Experiment Setup strip.
6. Results: main table + optional bar chart.
7. Case Analysis: six images.
8. Bottom left: Parameter sensitivity.
9. Bottom middle: Error taxonomy + what did not work.
10. Bottom right: Application + Conclusion + Future Work.
11. Footer: references, code/data, contact.

