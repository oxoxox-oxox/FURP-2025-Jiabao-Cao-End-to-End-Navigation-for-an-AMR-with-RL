# Poster Assets

This folder collects the files needed to revise and submit the STPS academic poster.

## Folder Structure

```text
poster_assets/
  figures/
    final_success_rate_bars.{png,svg}
    final_success_rate_avg.{png,svg}
    cases/
      success_u_trap.{png,svg}
      success_double_u.{png,svg}
      success_narrow_door.{png,svg}
      failure_baseline_u_trap.{png,svg}
      failure_explore_narrow_door.{png,svg}
      failure_neupan_compact.{png,svg}
  data/
    unified_comparison.json
    final_results_table.md
    full_rl_metrics.json
    stps_v2_results.json
    stps_v3_results.json
    stps_utrap_diagnosis.json
    final_evaluate_results.json
    expanded_eval_results.json
    *.csv
  scripts/
    generate_case_figures.py
    plot_final_bars.py
    README_final_stps_scripts.md
  text/
    poster_final_copy_paste.md
    poster_revision_content.md
    case_analysis_template.md
    submission_checklist.md
  source_poster/
    change.pptx
    change.pdf
    stps_poster_v2.pptx
    FURP_Showcase.tex
  from_downloads/
    source_poster/
      stps_poster_v3.pptx
      stps_poster_v4.pptx
      change.pptx
      change.pdf
    text/
      case_analysis.md
      references.bib
      experiment_summary.md
    figures/
      poster_barchart.png
      *.html
    paper_sections/
      method_section.tex
      experiments_section.tex
      related_work_section.tex
```

## What To Use In The Poster

### Main result table

Use:

- `data/final_results_table.md`
- source data: `data/unified_comparison.json`

Main result:

| Method | Standard | U-trap | Double-U | Narrow Door | Corridor | Scenario Avg. |
|---|---:|---:|---:|---:|---:|---:|
| CNNTD3 baseline | 87% | 0±0% | 69±4% | 100±0% | 100±0% | 67% |
| STPS v2 | 88% | 75±7% | 100±0% | 100±0% | 100±0% | 94% |

### Bar chart

Use one of:

- `figures/final_success_rate_bars.png`
- `figures/final_success_rate_avg.png`

The table is the primary result. The bar chart is optional if poster space is tight.

### Six case figures

Use all six:

- `figures/cases/failure_baseline_u_trap.png`
- `figures/cases/failure_explore_narrow_door.png`
- `figures/cases/failure_neupan_compact.png`
- `figures/cases/success_u_trap.png`
- `figures/cases/success_double_u.png`
- `figures/cases/success_narrow_door.png`

Caption summary:

- Failure: baseline cannot retreat in U-trap.
- Failure: exploration policy collides in narrow door.
- Failure: NeuPAN compact-scene configuration mismatch.
- Success: STPS switches to escape mode in U-trap.
- Success: STPS solves Double-U.
- Success: STPS preserves narrow-door precision.

### Full RL metrics

Use:

- `data/full_rl_metrics.json`

This supports the extra RL metrics required by the project brief:

- success rate;
- collision rate;
- average reward;
- average path length;
- path efficiency;
- average steps.

If poster space is limited, place these in the report/README instead of the poster.

## Copy-Paste Text

Use:

- `text/poster_final_copy_paste.md`

It contains ready-to-paste text for:

- Abstract;
- Introduction;
- Method;
- Experiment Setup;
- Results caption;
- Case analysis;
- Application;
- Conclusion;
- Future Work;
- Limitations;
- References.

## Re-generate Figures

Generate result bar charts:

```bash
cd ~/furp-2026-WenjingChen-RL-Navigation-for-AMR
python src/code/Wenjing_Chen/CNNTD3/final_stps/plot_final_bars.py
```

Generate trajectory/case figures:

```bash
conda activate neupan
cd ~/DRL-robot-navigation-IR-SIM
python ~/furp-2026-WenjingChen-RL-Navigation-for-AMR/src/code/Wenjing_Chen/CNNTD3/final_stps/generate_case_figures.py
```

After regenerating, copy updated files into `poster_assets/figures/` if needed.

## Source Poster Files

Use:

- `source_poster/change.pptx` as the latest editable PowerPoint source.
- `source_poster/change.pdf` as the latest reviewed PDF export.
- `source_poster/stps_poster_v2.pptx` as an earlier poster version.
- `from_downloads/source_poster/stps_poster_v4.pptx` if you want the latest PowerPoint found in Downloads.

Before final submission, export the final poster as:

```text
FURP_Showcase.pdf
```

and place it in the repository root.

## Final Poster Checklist

- [ ] Replace `[Author Name]`, `[University Name]`, and `[email]`.
- [ ] Add Abstract and Introduction text.
- [ ] Add Experiment Setup.
- [ ] Keep STPS flowchart.
- [ ] Include main result table.
- [ ] Include 3 success and 3 failure cases.
- [ ] Add Application.
- [ ] Expand Conclusion.
- [ ] Fill Future Work.
- [ ] Add a small References footer.
