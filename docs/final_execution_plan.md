# Final Execution Plan

This plan assumes the project is in its last week and should be frozen soon.

## Priority 1: Freeze the final story

Final story:

> STPS v2 is a runtime policy-switching method that preserves precision navigation while recovering from local minima.

Do not start a new method. Do not chase a new training branch.

## Priority 2: Collect the minimum missing evidence

### Must-have

1. **Six case images or GIF frames**
   - 3 success cases:
     - STPS U-trap
     - STPS Double-U
     - STPS narrow door
   - 3 failure cases:
     - CNNTD3 U-trap
     - exploration policy narrow door failure
     - NeuPAN compact-scene failure or table-based failure evidence

2. **One command block**
   - runnable baseline command;
   - final STPS evaluation command;
   - environment activation command.

3. **One final metrics table**
   - SR is the main metric;
   - reward/steps/collision are secondary;
   - if exact path efficiency is unavailable, state that explicitly.

### Nice-to-have

1. **Multi-seed repeat for the final evaluation**
   - If time allows, rerun baseline and STPS on the same hard-scenario set for 3 seeds.
   - This strengthens the claim without requiring new training.

2. **One short extra ablation**
   - STPS v2 vs STPS v3.
   - Or precision-only policy vs exploration-only policy.
   - Use this only if the result is already available.

3. **One small training-curve figure**
   - Use an existing TensorBoard or progress plot.
   - Do not retrain only to make a prettier curve.

## Priority 3: What to compute from existing logs

Use existing CSV/JSON logs to extract:

- success rate;
- timeout/collision counts;
- mean reward where available;
- mean step count.

If you cannot get shortest-path distance from the current logs, write:

> Exact path efficiency was not available in the current logging pipeline; average steps were used as the navigation-efficiency proxy.

## Priority 4: Poster and report ordering

### Poster order

1. Problem and baseline failure
2. Why a single policy is not enough
3. STPS method
4. Final result table
5. Three success and three failure cases
6. Limitations

### Report order

1. Task and setting
2. Baseline reproduction
3. Failure taxonomy
4. STPS method
5. Evaluation protocol
6. Results
7. What did not work
8. Reproducibility

## Priority 5: Submission package

You should submit:

- source code;
- environment and dependency notes;
- exact runnable commands;
- final evaluation table;
- 3 success and 3 failure cases;
- one focused improvement or ablation;
- a short failure summary;
- poster PDF as `FURP_Showcase.pdf`;
- final report if required by your supervisor;
- 5-8 minute presentation video if required by the project brief.

## Stop List

- No new model architecture.
- No new training branch unless it directly produces the final poster figure.
- No attempt to polish every forgotten detail.
- No invented numbers.

