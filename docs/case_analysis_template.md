# Success and Failure Case Analysis

This file gives a concise structure for the required 3 successful and 3 failed cases.

## How To Write Each Case

Use 5 lines per case:

- **Scenario:** where the robot starts, where the goal is, and what obstacle structure exists.
- **Method:** CNNTD3 baseline / exploration policy / STPS v2 / NeuPAN.
- **Observed behavior:** what the robot actually did.
- **Evidence:** SR, trajectory screenshot/GIF, collision, timeout, oscillation, or final distance.
- **Interpretation:** what this reveals about the policy.

Do not over-describe. A good case analysis is specific, short, and tied to the final claim.

## Three Successful Cases

### Success Case 1: STPS Escapes U-trap

- **Scenario:** U-trap hard scenario. The robot starts inside or near a concave U-shaped obstacle, while the goal direction points through the wall.
- **Method:** STPS v2.
- **Observed behavior:** The robot first behaves like the precision policy. After it stalls or oscillates, STPS switches to the exploration policy, moves out of the trap region, and then switches back to the precision policy to approach the goal.
- **Evidence:** STPS v2 reaches 75±7% SR on U-trap, while the CNNTD3 baseline remains 0±0%.
- **Interpretation:** The main baseline failure is not low-level collision avoidance, but lack of recovery behavior in local-minimum situations. Runtime switching adds this missing recovery behavior.

### Success Case 2: STPS Solves Double-U

- **Scenario:** Double-U hard scenario with two concave structures.
- **Method:** STPS v2.
- **Observed behavior:** The robot avoids getting permanently trapped in either U-shaped region and eventually reaches the goal.
- **Evidence:** STPS v2 reaches 100±0% SR, compared with CNNTD3 baseline 69±4%.
- **Interpretation:** STPS is not only tuned to one U-trap layout. The stall/oscillation trigger transfers to a more complex concave obstacle structure.

### Success Case 3: STPS Preserves Narrow-door Precision

- **Scenario:** Narrow-door hard scenario. The robot must align accurately and pass through a narrow opening.
- **Method:** STPS v2.
- **Observed behavior:** The robot passes through the door without unnecessary exploration behavior.
- **Evidence:** STPS v2 keeps 100±0% SR on narrow door, matching the CNNTD3 baseline.
- **Interpretation:** STPS does not sacrifice precision in scenes where the precision policy is already strong. This is important because exploration-heavy training previously damaged narrow-door performance.

## Three Failed Cases

### Failure Case 1: CNNTD3 Baseline Freezes in U-trap

- **Scenario:** U-trap hard scenario.
- **Method:** CNNTD3 baseline.
- **Observed behavior:** The robot remains inside the U-shaped region, freezes, or oscillates locally instead of backing out.
- **Evidence:** CNNTD3 baseline has 0±0% SR on U-trap.
- **Interpretation:** A reactive policy trained mostly on standard navigation does not learn backtracking. The goal signal points toward the wall, creating a local-minimum failure.

### Failure Case 2: Single Exploration-trained Policy Loses Narrow-door Precision

- **Scenario:** Narrow-door hard scenario after training with curriculum + exploration reward.
- **Method:** Curriculum + exploration policy.
- **Observed behavior:** The policy becomes more willing to move and escape traps, but it loses precise alignment behavior at the doorway.
- **Evidence:** Earlier ablation showed U-trap improved to 100%, while narrow-door success dropped to 0%.
- **Interpretation:** Exploration and precision are in tension. This directly motivates switching between policies instead of forcing one policy to keep both behaviors.

### Failure Case 3: NeuPAN Configuration Mismatch in Compact Benchmark

- **Scenario:** Compact 10 x 10 m hard-scenario benchmark.
- **Method:** NeuPAN under the tested forward-only / safety-margin configuration.
- **Observed behavior:** The planner fails to make progress across the compact hard scenarios.
- **Evidence:** In the unified comparison, NeuPAN has 0% SR on U-trap, Double-U, narrow door, and corridor.
- **Interpretation:** This should be reported carefully as a domain/configuration mismatch. It is useful as a negative comparison, but not the main final claim.

## Error Taxonomy

Use this taxonomy in the report/poster:

| Error Type | Example | Cause | Possible Fix |
|---|---|---|---|
| Local-minimum trap | CNNTD3 in U-trap | goal direction points through obstacle | stall-triggered escape policy |
| Precision failure | exploration policy in narrow door | exploration behavior hurts alignment | switch back to precision policy |
| Oscillation/stall | repeated small movements | policy uncertainty near obstacle boundary | oscillation trigger |
| Configuration mismatch | NeuPAN in compact scenes | safety margin / forward-only constraint too conservative | retune planner or robot model |

## Figure Checklist

Best case:

- one U-trap success trajectory for STPS;
- one Double-U success trajectory for STPS;
- one narrow-door success trajectory for STPS;
- one U-trap failure trajectory for CNNTD3;
- one narrow-door failure trajectory for exploration policy;
- one NeuPAN failure screenshot or table-only failure evidence.

Acceptable fallback:

- If some trajectory images are missing, use scenario diagrams plus the result table, and clearly label them as representative scenario layouts.

