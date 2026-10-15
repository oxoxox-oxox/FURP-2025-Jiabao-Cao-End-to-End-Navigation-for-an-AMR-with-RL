"""Recompute archived success statistics without simulator or model weights."""
import argparse
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ('S1_U_trap', 'S2_Double_U', 'S3_Narrow_door', 'S5_Corridor')


def summarize(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    averages = {}
    for method, scenes in data.items():
        values = []
        for scenario in SCENARIOS:
            entry = scenes[scenario]
            seeds = entry['seeds']
            if len(seeds) != 3 or any(not isinstance(x, (int, float)) or
                    not math.isfinite(x) or not 0 <= x <= 1 for x in seeds):
                raise ValueError(f'{method}/{scenario}: expected three finite success rates in [0,1]')
            mean = statistics.mean(seeds)
            std = statistics.pstdev(seeds)
            if abs(mean-entry['mean']) > .00051 or abs(std-entry['std']) > .00051:
                raise ValueError(f'{method}/{scenario}: archived aggregates disagree with seed rates')
            values.append(mean)
        averages[method] = statistics.mean(values)
    return averages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, default=ROOT / 'src/results/final/unified_comparison.json')
    args = parser.parse_args()
    for method, average in summarize(args.file).items():
        print(f'{method}: hard-scenario mean SR={average:.2%}')
    print('Archived statistics verified; no policy evaluation was rerun.')


if __name__ == '__main__':
    main()
