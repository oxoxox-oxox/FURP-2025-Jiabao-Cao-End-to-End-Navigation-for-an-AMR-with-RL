# Final Week Recovery Plan

This document is a practical recovery plan for the final week. The goal is not to add a new research direction. The goal is to turn the existing experiments into a clear, reproducible story.

## 1. Final Story

Use this as the final project title:

**Escaping Local Minima in End-to-End AMR Navigation with Stall-Triggered Policy Switching**

One-sentence story:

> A single CNNTD3 navigation policy struggled to keep both precise navigation and trap-escape behavior, so I introduced STPS, a runtime policy-switching method triggered by stagnation and oscillation, which improved hard-scenario average success from 67% to 94% while preserving standard-environment performance.

Main contribution:

- Not a new SOTA model.
- Not a full paper.
- A focused, reproducible system improvement over a reproduced CNNTD3 baseline.

Do not make NeuPAN delay compensation the main story unless more real-robot evidence is finished. Keep NeuPAN as a secondary comparison / negative result.

## 2. What To Submit

Minimum final submission package:

- `README.md`: project overview, setup, commands, final result table.
- `docs/00_weekly.md`: weekly progress log, already recovered.
- `docs/final_demo_script.md`: 5-8 minute video script.
- Source code:
  - CNNTD3 baseline evaluation.
  - STPS evaluation.
  - Sensitivity test if available.
- Results:
  - `unified_comparison.json`
  - `stps_sensitivity_results.json`
  - one final CSV or markdown table copied into the report.
- Visuals:
  - at least 3 successful trajectories.
  - at least 3 failed trajectories.
  - one comparison plot/table.
- Final report or poster:
  - problem;
  - baseline;
  - failure analysis;
  - method;
  - results;
  - limitations.
- 5-8 minute video.

## 3. Data Organization

Create or keep this structure:

```text
src/results/final/
  unified_comparison.json
  stps_sensitivity_results.json
  final_results_table.md
  success_cases/
  failure_cases/
```

If the original files are still outside this repository, copy only the final JSON/CSV/tables and representative media into the repo. Avoid moving entire training folders unless they are small and necessary.

Use this final result table in the report and video:

| Method | Standard | U-trap | Double-U | Narrow Door | Corridor | Scenario Avg. |
|---|---:|---:|---:|---:|---:|---:|
| CNNTD3 baseline | 87% | 0±0% | 69±4% | 100±0% | 100±0% | 67% |
| NeuPAN | 0% | 0±0% | 0±0% | 0±0% | 0±0% | 0% |
| STPS v2 | 88% | 75±7% | 100±0% | 100±0% | 100±0% | 94% |

Be honest about the NeuPAN row:

> In my compact 10 x 10 m evaluation setting, NeuPAN failed under the forward-only and safety-margin configuration. This is a configuration/domain mismatch result, not a claim that NeuPAN generally fails.

## 4. How To Recover Forgotten Details

Do not rely only on memory. Recover evidence in this order:

1. Use file modification times:
   - scripts;
   - JSON/CSV results;
   - log files;
   - generated GIFs or screenshots.
2. Search terminal history for commands:
   - `history | grep stps`
   - `history | grep eval`
   - `history | grep neupan`
3. Search filenames:
   - `find ~/DRL-robot-navigation-IR-SIM -iname '*stps*'`
   - `find ~/NeuPAN -iname '*delay*'`
4. Open result JSON/CSV files and extract only:
   - method name;
   - scenario;
   - seed count;
   - episode count;
   - SR / collision / timeout / path length if available.
5. If a detail cannot be recovered, write:
   - "not fully recorded";
   - "not used in final comparison";
   - "kept as exploratory result".

Do not invent exact values.

## 5. Final Week Schedule

### Day 1: Freeze the story

- Finalize the title and contribution.
- Stop new training.
- Collect final result files.
- Decide which 3 successes and 3 failures will be shown.

### Day 2: Reproducibility cleanup

- Write the exact commands for:
  - baseline evaluation;
  - STPS evaluation;
  - sensitivity test.
- Record environment:
  - OS;
  - Python;
  - PyTorch;
  - IR-SIM;
  - CUDA/GPU if relevant.

### Day 3: Figures and cases

- Make one final result table.
- Make one sensitivity table or heatmap.
- Export 3 success and 3 failure trajectory images/GIFs.

### Day 4: Report/poster

Use this section order:

1. Problem and task
2. Baseline reproduction
3. Failure diagnosis
4. STPS method
5. Evaluation protocol
6. Results
7. Limitations
8. Reproducibility

### Day 5: Video

- Record screen with slides and trajectory clips.
- Keep it between 5 and 8 minutes.
- Use the script in `docs/final_demo_script.md`.

### Day 6-7: Buffer

- Fix broken paths.
- Check that videos/images open.
- Check that commands are written clearly.
- Push or package the final repository.

## 6. What Not To Do

- Do not start a new model.
- Do not try to make NeuPAN delay compensation into the main contribution now.
- Do not claim paper-level novelty.
- Do not hide failed attempts; use them to explain why the final choice is reasonable.
- Do not chase missing old details if they are not needed for the final story.

## 7. Final Claim

Use a modest final claim:

> STPS is a lightweight runtime policy-switching strategy that uses stagnation and oscillation evidence to select between a precision-oriented policy and an exploration-oriented policy. In the tested hard-scenario benchmark, it preserved standard navigation success and substantially improved trap-escape performance.

