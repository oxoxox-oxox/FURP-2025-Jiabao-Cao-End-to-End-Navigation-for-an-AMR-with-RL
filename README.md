[English](README.md) · [简体中文](README.zh-CN.md)

# STPS: Escaping Navigation Traps Without Losing Precision

**A FURP team project on end-to-end mobile robot navigation with reinforcement learning.**

**[RESEARCH POSTER — PDF](FURP_Showcase.pdf) · [Presentation — PPTX](FURP_Showcase.pptx)**

<a href="FURP_Showcase.pdf"><img src="docs/assets/stps-poster.png" alt="STPS research poster" width="600"></a>

## Value → contribution → method

A robot that always moves toward its goal can remain trapped behind concave obstacles. Exploration helps it escape, but can destroy narrow-door precision. **Stall-Triggered Policy Switching (STPS)** combines complementary trained policies at runtime to address this trade-off.

The team's contribution is a stall/oscillation detector, a precision-to-exploration switching controller, and a matched benchmark across four hard scenarios. The repository includes CNN-TD3 models, simulation interfaces, training/evaluation scripts and per-seed result summaries. This demonstrates policy composition, continuous robot control and experimental comparison.

**Zexian Xiong (Sean)** is a team member and poster co-author; Git history also records his repository documentation contribution (`6b8eee5`). Individual ownership of the algorithm and experiments is not broken down in the available records, so the research contributions below are credited to the team.

**Final method: STPS v2.** It triggers escape when displacement is below 0.15 m over a 20-position window, or at least 5 direction reversals occur in 12 positions. It returns after a minimum escape duration and displacement above 0.5 m, with a timeout fallback. It uses frozen policies and requires no additional training for the switch. V3 adds goal-distance detection and cooldown; it is an exploratory variant, not the final result.

## Results and evidence

| Method | U-trap | Double-U | Narrow door | Corridor | Hard-scenario mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN-TD3 baseline | 0% | 69±4% | 100% | 100% | 67.36% |
| **STPS v2** | **75±7%** | **100%** | **100%** | **100%** | **93.75%** |

From [archived per-seed results](src/results/final/unified_comparison.json): 3 seeds × 12 perturbed starts per hard scenario; ± is population standard deviation across seed success rates. The mean gain is **26.39 percentage points**; the poster rounds averages to 94% versus 67%. The [final table](src/results/final/final_results_table.md) separately reports standard-environment SR 88% versus 87% from 100 episodes; episode logs for that claim are absent.

These statistics are **recomputed from archived summaries**, not fresh trained-policy runs. The trained checkpoints are missing. Runtime overhead and real-robot readiness claims in the poster were not independently tested here. [Evidence and limitations](docs/REPRODUCIBILITY.md) explain what can be reproduced.

## Quick start

Recompute archived statistics with Python only; no model, GPU or account is needed:

```sh
git clone https://github.com/oxoxox-oxox/FURP-2025-Jiabao-Cao-End-to-End-Navigation-for-an-AMR-with-RL.git
cd FURP-2025-Jiabao-Cao-End-to-End-Navigation-for-an-AMR-with-RL
python tools/summarize_results.py
```

Expected: `CNNTD3_baseline ... 67.36%`, `NeuPAN ... 0.00%`, `STPS_v2 ... 93.75%`. The NeuPAN value is limited to this archived compact-scene configuration and is not a general conclusion about the method.

For simulator, switching and model interface checks, use Python 3.12:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux shell instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -X utf8 tools/verify_project.py
```

Success ends with `VERIFICATION PASSED`. The real IR-SIM smoke test runs the original switching function in four worlds with synthetic stationary/rotating policies: 19 main-policy calls, 41 escape calls and timeout per scene. A separate check runs the original CNN-TD3 actor/critics with random initial weights. **Neither measures learned navigation performance.** Generated outputs stay in ignored `artifacts/`.

Installation needs internet and substantial space for CPU PyTorch; verification itself is offline, takes seconds to tens of seconds and uses no paid service. No credentials are needed. See [complete reproduction instructions](docs/REPRODUCIBILITY.md) for pinned dependencies, commands and missing-resource checks.

## Structure, development and status

`src/code/robot_nav/worlds/` defines scenes; `sim/` turns LiDAR, goal geometry and previous actions into observations. `models/cnntd3.py` encodes **180 LiDAR beams + 5 goal/action values = 185 inputs**, producing two velocity actions. `scripts/test/eval_unified.py` selects precision/escape policies, executes episodes and aggregates success rates. `src/results/final/` preserves archived results; `docs/` retains research history. [Source map](src/code/README.md).

The repository is a research snapshot. Full trained STPS inference is blocked by 12 missing checkpoint files; `python tools/check_resources.py --strict` lists them and exits 2. Historical training and hardware workflows are not verified. Core code and data remain unchanged. [Validation record](docs/VALIDATION.md).

Run `python -X utf8 -m unittest discover -s tests -v`; CI additionally exercises the simulator and CPU model. Report issues through [GitHub Issues](https://github.com/oxoxox-oxox/FURP-2025-Jiabao-Cao-End-to-End-Navigation-for-an-AMR-with-RL/issues), including commands and environment without secrets. Request permission before reuse or modification.

**All rights reserved; no open-source license is granted.** Public visibility is for platform viewing; other use, redistribution or modification requires permission from the relevant rights holders. Team authors, supervisors and external software retain their attribution and rights. [Rights and credits](RIGHTS.md).
