# Final Demo Video Script

Target length: 5-8 minutes.

Main rule: show evidence, not ambition. The video only needs to prove that the project is reproducible, evaluated, and understood.

## 0:00-0:40 Opening

Say:

> My project is about end-to-end navigation for an autonomous mobile robot. The agent receives local observations and must reach the goal while avoiding obstacles. My final focus is not visual-language navigation, but reinforcement-learning-based AMR navigation and failure recovery.

Show:

- One screenshot or GIF of the robot navigating.
- Project title:
  **Escaping Local Minima in End-to-End AMR Navigation with Stall-Triggered Policy Switching**

## 0:40-1:30 Baseline

Say:

> I reproduced a CNNTD3 navigation baseline in IR-SIM. In the standard environment, the independent 100-episode success rate was 87%. However, when I evaluated the same policy in structured hard scenarios, several failure modes became clear.

Show:

- CNNTD3 architecture or code entry point.
- Standard-environment evaluation command if available.
- Baseline table row.

Mention:

- Simulator: IR-SIM.
- Policy: CNNTD3.
- Evaluation: standard scene and hard scenarios.

## 1:30-2:40 Failure Diagnosis

Say:

> The main failures were not random. They followed recognizable patterns: concave traps, double-U structures, and precision bottlenecks such as narrow doors. The U-trap is especially important because the goal direction points through a wall, so a purely reactive policy tends to freeze or oscillate instead of backing out.

Show:

- U-trap failed trajectory.
- Double-U failed trajectory.
- Narrow-door or corridor example.

Use this taxonomy:

| Failure Type | Evidence | Interpretation |
|---|---|---|
| Concave trap | U-trap SR 0% | Policy lacks backtracking behavior |
| Precision bottleneck | Narrow door | Policy needs accurate alignment |
| Oscillation/stall | repeated turns or small displacement | Policy is uncertain but has no recovery mode |

## 2:40-3:40 Failed Training Attempts

Say:

> I tried to solve this through training. Exploration reward helped the U-trap, but it hurt narrow-door precision. Annealing the exploration reward recovered narrow-door performance, but U-trap success dropped back to 0%. This suggested that the two behaviors were difficult to keep in a single policy.

Show:

- Ablation table:

| Scenario | Baseline | Curriculum + Exploration |
|---|---:|---:|
| U-trap | 0% | 100% |
| Narrow door | 4.8% | 0% |

- Mention the annealing result:
  - narrow door: 100%;
  - U-trap: 0%.

## 3:40-4:50 Method: STPS

Say:

> Based on this, I implemented STPS: Stall-Triggered Policy Switching. Instead of forcing one policy to do everything, the robot normally uses a precision policy. When it detects stagnation or oscillation, it switches to an exploration policy for a short escape phase, then switches back after it has moved away.

Show:

- Simple flow diagram or slide:

```text
precision policy
  -> stall or oscillation detected
  -> exploration policy for escape
  -> displacement recovered
  -> precision policy
```

Mention parameters:

- stall: 20-step displacement < 0.15 m;
- oscillation: at least 5 direction reversals in 12 steps;
- escape duration: 120 steps;
- recovery: displacement > 0.5 m.

## 4:50-6:10 Results

Say:

> STPS kept standard-environment performance almost unchanged, from 87% to 88%, while improving the hard-scenario average from 67% to 94%. The largest gains were in U-trap and Double-U scenarios.

Show:

| Method | Standard | U-trap | Double-U | Narrow Door | Corridor | Scenario Avg. |
|---|---:|---:|---:|---:|---:|---:|
| CNNTD3 baseline | 87% | 0±0% | 69±4% | 100±0% | 100±0% | 67% |
| STPS v2 | 88% | 75±7% | 100±0% | 100±0% | 100±0% | 94% |

Then show:

- 1 STPS U-trap success clip.
- 1 Double-U success clip.
- 1 narrow-door success clip to show precision is preserved.

## 6:10-7:10 Secondary Experiments

Say:

> I also explored NeuPAN and delay compensation. These experiments were useful, but I do not use them as the main contribution. NeuPAN had configuration mismatch issues in my compact 10 x 10 m scenes, and delay compensation likely needs stronger real-robot evidence. I therefore keep them as exploratory results and focus the final claim on STPS.

Show:

- One NeuPAN/Delay slide only.
- Do not spend too long here.

## 7:10-7:50 Limitations and Conclusion

Say:

> The limitation is that STPS depends on hand-designed switching thresholds and was tested in a limited simulator benchmark. It is not a universal navigation solution. However, the project gives a reproducible pipeline: baseline reproduction, hard-scenario diagnosis, a focused improvement, and fair comparison.

End with:

> My main takeaway is that failure analysis was more useful than blindly training larger policies. By identifying the conflict between exploration and precision, I could design a small runtime mechanism that improved the cases where the baseline failed.

## Recording Checklist

- Keep the video between 5 and 8 minutes.
- Use one slide per section.
- Show at least one real trajectory clip or GIF.
- Show the final table clearly.
- Do not apologize for not publishing a paper.
- Say "exploratory result" for unfinished branches.
- Say "not fully recorded" if a detail is genuinely missing.

