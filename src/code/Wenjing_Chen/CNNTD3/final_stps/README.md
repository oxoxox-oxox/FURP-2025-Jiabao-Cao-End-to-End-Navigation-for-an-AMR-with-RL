# Final STPS Scripts

This folder contains the final STPS evaluation scripts copied from the working experiment directory.

## Files

- `eval_stps_v2.py`
  - Final STPS v2 evaluation script.
  - Uses stall detection and oscillation detection to switch between the main policy and the escape policy.
- `eval_unified.py`
  - Unified comparison script used for final hard-scenario comparison.

## Final Method

Use `eval_stps_v2.py` as the implementation evidence for the final demo/report.

Important STPS v2 parameters:

- `STALL_WINDOW = 20`
- `STALL_DIST = 0.15`
- `BASE_ESCAPE_STEPS = 120`
- `PROGRESS_DIST = 0.5`
- `OSC_WINDOW = 12`
- `OSC_REVERSAL_THRESH = 5`

## Runnable Commands

```bash
conda activate <your_env>
cd ~/DRL-robot-navigation-IR-SIM
python eval_unified.py
python eval_stps_v2.py
```

Expected outputs:

- `unified_comparison.json`
- `stps_v2_results.json`
- `stps_v3_results.json` if you run the exploratory variant

## Note

These scripts may still depend on the original working directory layout under `~/DRL-robot-navigation-IR-SIM`. For final submission, report the exact working path and environment rather than claiming the scripts are fully standalone.
