# Final Result Files

This folder contains the evidence used for the final STPS story.

## Main Files

- `final_results_table.md`
  - Human-readable final result table for report, poster, and demo video.
- `unified_comparison.json`
  - Main hard-scenario comparison among CNNTD3 baseline, NeuPAN, and STPS v2.
- `stps_v2_results.json`
  - Detailed STPS v2 output.
- `stps_v3_results.json`
  - Exploratory STPS v3 output. It did not improve over v2 and is not used as the final method.
- `stps_utrap_diagnosis.json`
  - U-trap diagnosis evidence.

## Final Method

Use **STPS v2** as the final method.

Final claim:

> STPS v2 improves hard-scenario navigation by switching from a precision policy to an exploration policy when stagnation or oscillation is detected, then switching back after escape progress is made.

## Reporting Caution

- Do not present STPS v3 as the final method.
- Do not overclaim NeuPAN failure. The NeuPAN result reflects the compact benchmark and tested configuration.
- If an experiment detail cannot be recovered from logs or files, mark it as exploratory or not fully recorded.

