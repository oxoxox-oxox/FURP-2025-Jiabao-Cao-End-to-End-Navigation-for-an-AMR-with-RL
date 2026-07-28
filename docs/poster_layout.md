# Poster Layout

Recommended poster title:

**Escaping Local Minima in End-to-End AMR Navigation with Stall-Triggered Policy Switching**

## Main Message

Put this near the title:

> A CNNTD3 navigation policy can be precise in narrow passages or exploratory in traps, but one policy struggled to keep both abilities. STPS switches policies only when stall or oscillation is detected, improving hard-scenario average success from 67% to 94% while preserving standard-environment performance.

## Poster Structure

Use a 3-column layout.

### Column 1: Problem and Baseline

Sections:

1. **Task**
   - End-to-end AMR navigation in IR-SIM.
   - Input: local observation / LiDAR-like state.
   - Output: continuous navigation action.

2. **Baseline**
   - CNNTD3 baseline.
   - Standard environment SR: 87%.
   - Hard scenarios: U-trap, Double-U, narrow door, symmetric corridor.

3. **Failure Diagnosis**
   - U-trap: 0±0% SR.
   - Baseline stalls/oscillates because the goal direction points through the wall.
   - Exploration reward helps traps but hurts narrow-door precision.

Suggested visual:

- one hard-scenario layout figure;
- one failed U-trap trajectory;
- small failure taxonomy table.

### Column 2: Method

Section title:

**Stall-Triggered Policy Switching**

Explain STPS in a simple flow:

```text
Precision policy
  -> detect stall or oscillation
  -> switch to exploration policy
  -> escape for 120 steps
  -> switch back after progress
```

Parameters:

| Trigger | Setting |
|---|---|
| Stall | 20-step displacement < 0.15 m |
| Oscillation | >=5 direction reversals in 12 steps |
| Escape duration | 120 steps |
| Recovery | displacement > 0.5 m |

Suggested visual:

- flow diagram;
- one STPS success trajectory in U-trap;
- one STPS success trajectory in narrow door showing precision is preserved.

### Column 3: Results and Takeaways

Main result table:

| Method | Standard | U-trap | Double-U | Narrow Door | Corridor | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| CNNTD3 | 87% | 0±0% | 69±4% | 100±0% | 100±0% | 67% |
| NeuPAN | 0% | 0±0% | 0±0% | 0±0% | 0±0% | 0% |
| STPS v2 | 88% | 75±7% | 100±0% | 100±0% | 100±0% | 94% |

Takeaways:

- STPS improves trap and Double-U performance.
- STPS preserves standard and narrow-door performance.
- Failure analysis was more useful than adding another training run.
- NeuPAN result is treated as a compact-scene configuration mismatch, not a general claim.

Limitations:

- Thresholds are hand-designed.
- Evaluation is simulator-only.
- Real-robot delay compensation remains exploratory.

## What To Avoid

- Do not include every week of experiments.
- Do not make NeuPAN delay compensation a main section.
- Do not put large code blocks on the poster.
- Do not claim SOTA or paper-level novelty.
- Do not use more than one main table.

## Minimum Visual Set

Enough for the poster:

- 1 scenario-layout figure.
- 1 baseline failure trajectory.
- 2 STPS success trajectories.
- 1 final result table.
- 1 small STPS flow diagram.

If only one GIF is available, use still frames or screenshots from it.

