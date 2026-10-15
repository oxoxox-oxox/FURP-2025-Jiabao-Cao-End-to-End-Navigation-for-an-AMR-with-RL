"""Exercise the original STPS v2 in IR-SIM with explicit synthetic policies."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
from stps_source import CODE, ROOT, load_definitions
import numpy as np
sys.path.insert(0, str(CODE))
from robot_nav.sim.sim import SIM


class ConstantPolicy:
    def __init__(self, action):
        self.action = np.array(action, dtype=float)
        self.calls = 0

    def prepare_state(self, *observation):
        return np.zeros(185), False

    def get_action(self, state, add_noise):
        self.calls += 1
        return self.action.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/smoke_stps.json')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    namespace = load_definitions(SIM)
    results = {}
    for name, scene in namespace['SCENARIOS'].items():
        # Stationary main and rotating escape deliberately exercise the stall branch.
        main_policy = ConstantPolicy([-1, 0])
        escape_policy = ConstantPolicy([-1, 1])
        outcome = namespace['run_stps_v2'](
            main_policy, escape_policy, str(CODE / scene['world']),
            [[scene['base_xy'][0]], [scene['base_xy'][1]], [0.0]], scene['goal'], 60)
        assert main_policy.calls == 19, (name, main_policy.calls)
        assert escape_policy.calls == 41, (name, escape_policy.calls)
        assert outcome == 'timeout', (name, outcome)
        results[name] = {'outcome': outcome, 'main_calls': main_policy.calls,
                         'escape_calls': escape_policy.calls}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'seed': args.seed, 'synthetic_policies': True,
                                     'scenarios': results}, indent=2)+'\n', encoding='utf-8')
    print('STPS SMOKE PASSED: four worlds, original switching code, main=19 / escape=41 per world.')
    print('Synthetic policies only. This does not measure trained STPS success rates.')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
