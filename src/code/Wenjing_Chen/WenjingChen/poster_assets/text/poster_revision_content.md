# Poster Revision Content

This file gives paste-ready text for improving `change.pdf` / `change.pptx`.

## 0. Title Area

Current subtitle:

> Composing complementary policies at runtime to eliminate catastrophic navigation failures

Safer academic subtitle:

> Runtime composition of complementary DRL policies for reducing catastrophic failures in trap-like navigation scenarios

Replace placeholder author line:

> Wenjing Chen · University of Nottingham Ningbo China · Summer Research 2026

Footer:

> Code and data available · Contact: [your email] · Summer Research Program 2026

Metric card wording:

- `94% Scenario avg SR (+27 pp vs baseline)`
- `88% Standard env SR (matches baseline)`
- `0 hard-scenario 0%-SR cells`
- `0 additional training cost`

Avoid saying simply `0 catastrophic failures`, because it sounds like every episode succeeds.

## 1. Abstract

Use this paragraph:

> Deep reinforcement learning (DRL) enables mapless mobile robot navigation from local sensory observations, but reactive policies can still fail catastrophically in trap-like layouts where reaching the goal requires temporarily moving away from it. In our IR-SIM benchmark, a CNNTD3 baseline achieves 87% success in the standard environment but 0% in the U-trap scenario. We find an exploration-precision conflict: exploration reward improves U-trap escape but destroys narrow-door precision, while annealing restores precision but loses escape behavior. To address this, we propose Stall-Triggered Policy Switching (STPS), a training-free runtime mechanism that switches between a precision policy and an exploration policy only when stagnation or oscillation is detected. Across four hard scenarios and three seeds, STPS improves scenario-average success from 67% to 94% while preserving standard-environment performance at 88%.

If space is tight, use this shorter version:

> DRL navigation policies can succeed in standard scenes but fail catastrophically in trap-like layouts. We identify an exploration-precision conflict: exploration helps U-trap escape but harms narrow-door alignment. STPS addresses this by switching between frozen precision and exploration policies at runtime when stall or oscillation is detected. On four hard scenarios, STPS improves average success from 67% to 94% without additional training.

## 2. Introduction

Add below Abstract or replace some blank space under Abstract:

> Mapless DRL navigation maps local observations, such as LiDAR and relative goal state, directly to continuous robot actions. This is attractive for AMRs because it avoids explicit map building and can react online in unknown environments. However, purely reactive policies often lack long-horizon recovery behavior. In concave traps, the goal direction may point through an obstacle, so greedy goal tracking causes freezing, oscillation, or timeout. This project asks whether a lightweight runtime mechanism can recover from such local minima without sacrificing precision in narrow passages.

Three bullet version:

- Mapless DRL is attractive for AMRs because it avoids explicit maps and outputs continuous actions directly.
- Trap-like layouts expose a local-minimum failure: the robot must temporarily retreat from the goal.
- We focus on runtime recovery rather than training a larger model.

## 3. Key Finding

Your current section is good. Add one sentence at the bottom:

> This suggests the failure is not only about model capacity, but about conflicting behavioral modes: exploration for escape and precision for alignment.

## 4. Method: STPS Architecture

Your current flowchart is good. Replace `Real-robot ready` with:

> Robot-deployment friendly

or:

> Simple deployment logic

Add one compact algorithm box:

```text
At each step:
1. Run precision policy by default.
2. If displacement over 20 steps < 0.15 m, or direction reversals >= 5/12 steps:
   switch to exploration policy.
3. Keep escape mode for at least 120 steps.
4. Switch back when displacement from switch point > 0.5 m.
```

## 5. Experiment Setup

Place immediately before Results. Use compact bullets:

> **Environment:** IR-SIM, 10 m x 10 m 2D indoor navigation worlds, differential-drive robot, 180-beam LiDAR, continuous linear/angular velocity control.

> **Scenarios:** one standard random-obstacle environment plus four structured hard scenarios: U-trap, Double-U, narrow door, and symmetric corridor.

> **Baselines:** CNNTD3 baseline, RCPG (GRU), curriculum-only policy, exploration policy, annealed precision policy, and NeuPAN configuration test.

> **Evaluation:** hard scenarios use 3 seeds x 12 perturbed starts per scenario; standard environment uses 100 episodes. Primary metric is Success Rate (SR). Collision/timeout outcome and average steps are used as secondary diagnostics.

> **STPS policies:** both component policies are frozen. STPS adds only runtime switching logic and no additional training.

If you want a table:

| Item | Setting |
|---|---|
| Simulator | IR-SIM |
| Robot | differential-drive AMR |
| Sensor | 180-beam 2D LiDAR |
| Action | continuous linear/angular velocity |
| Hard scenarios | U-trap, Double-U, narrow door, corridor |
| Evaluation | 3 seeds x 12 starts; standard 100 episodes |
| Main metric | Success Rate (SR) |

## 6. Results Caption

Add under the result table:

> STPS is the only tested method that removes all 0%-SR hard-scenario cells while preserving standard-environment success. The gain is largest in concave-trap scenarios, where switching introduces temporary retreat behavior absent from the precision-only baseline.

## 7. Case Analysis Caption

Use these labels:

- Failure: baseline cannot retreat in U-trap.
- Failure: exploration policy collides in narrow door.
- Failure: annealed policy recovers precision but loses U-trap escape.
- Success: STPS switches to escape mode in U-trap.
- Success: STPS switches back and preserves narrow-door precision.
- Success: STPS keeps standard navigation performance.

Add this one-line summary:

> The cases show that STPS does not simply make the robot more exploratory; it activates exploration only when recovery evidence is observed.

## 8. Parameter Sensitivity Caption

Add under the sensitivity table:

> Narrow-door SR remains 100% across tested thresholds, indicating that switching does not harm precision. U-trap performance is most stable around a 20-step stall window, suggesting that too-short windows trigger prematurely while too-long windows delay recovery.

## 9. Error Taxonomy and What Did Not Work

Your current section is good. Add:

> These negative results motivate STPS: rather than merging all behavior into one policy, the controller preserves specialized policies and composes them conditionally.

## 10. Application

Add a new small section, preferably near Future Work or bottom-right:

> **Application Potential**
>
> STPS is suitable for AMR navigation in warehouses, corridors, offices, and service-robot scenarios where robots may encounter cul-de-sacs, narrow passages, or temporary local minima. Because STPS uses frozen policies and simple state-history checks, it can be integrated as a lightweight supervisory layer above an existing learned controller. The method is also attractive for sim-to-real transfer because the switching logic does not require differentiable training or GPU inference beyond the existing policy.

Shorter version:

- Lightweight supervisor for AMR local navigation.
- Useful in warehouses, offices, corridors, and narrow-passage layouts.
- Training-free switching layer can wrap existing DRL controllers.
- Sim-to-real friendly because switching uses only recent odometry/position history.

## 11. Conclusion

Replace current conclusion with:

1. We reproduce and diagnose a CNNTD3 navigation baseline that performs well in standard scenes but fails catastrophically in trap-like layouts.
2. We identify an exploration-precision conflict: exploration helps local-minimum escape but damages narrow-door alignment.
3. STPS resolves this conflict by composing two frozen policies at runtime using stall and oscillation evidence.
4. In our hard-scenario benchmark, STPS improves scenario-average SR from 67% to 94% and removes all 0%-SR hard-scenario cells while preserving standard-environment performance.

## 12. Future Work

Replace the current empty item with:

1. Real-robot validation on TurtleBot/LIMO platform with odometry-based stall detection.
2. Learned switching trigger using trajectory history or a lightweight scene classifier.
3. Dynamic-obstacle extension with delay-aware obstacle prediction.
4. Broader evaluation on randomized trap geometries and unseen narrow passages.
5. Integration with ROS 2 Nav2 as a recovery behavior or local-controller supervisor.

## 13. Limitations

Optional but useful:

> **Limitations:** STPS currently uses hand-designed thresholds and is evaluated mainly in 2D simulation. Exact shortest-path efficiency is not fully logged in all runs, and real-robot robustness remains future work. NeuPAN results should be interpreted as compact-benchmark configuration mismatch rather than a general failure of model-based planners.

## 14. Related Work / References

Use a small footer references block:

1. Tai, L., Paolo, G., & Liu, M. Virtual-to-real deep reinforcement learning: continuous control of mobile robots for mapless navigation. arXiv:1703.00420, 2017.
2. Fujimoto, S., van Hoof, H., & Meger, D. Addressing function approximation error in actor-critic methods. ICML, 2018.
3. Liang, J. et al. Context-Aware Deep Reinforcement Learning for Autonomous Robotic Navigation in Unknown Area. CoRL, 2023.
4. Hu, Y. et al. Deep reinforcement learning-based mapless navigation for mobile robot in unknown environment with local optima. IEEE RA-L, 2024.
5. Han, R. et al. NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning. TRO, 2025.
6. Khatib, O. Real-time obstacle avoidance for manipulators and mobile robots. IJRR, 1986.

## 15. Layout Recommendation

Current poster has too much blank space under Abstract and Future Work. Suggested edits:

- Put Abstract + Introduction in the upper-left block.
- Put Experiment Setup as a thin block between Method and Results.
- Keep the main Results table; the bar chart can be smaller or optional.
- Keep all six case images, but use one-line captions.
- Expand Future Work and add Application in the bottom-right.
- Add a small References line in the footer.

