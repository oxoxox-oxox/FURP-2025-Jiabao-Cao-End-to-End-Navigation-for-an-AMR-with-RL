"""Report exact trained checkpoint files expected by the original v2 evaluator."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ('CNNTD3', 'CNNTD3_v7_finetune_best', 'CNNTD3_improved')
SUFFIXES = ('actor', 'actor_target', 'critic', 'critic_target')


def missing_checkpoints():
    folder = ROOT / 'src/code/robot_nav/models/CNNTD3/checkpoint'
    return [folder / f'{prefix}_{suffix}.pth' for prefix in PREFIXES for suffix in SUFFIXES
            if not (folder / f'{prefix}_{suffix}.pth').is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true', help='Exit 2 if any checkpoint is absent.')
    args = parser.parse_args()
    missing = missing_checkpoints()
    for path in missing:
        print(f'MISSING {path.relative_to(ROOT).as_posix()}')
    print(f'{len(missing)} / 12 required checkpoint files missing.')
    if missing:
        print('Full trained-policy benchmark is unavailable; archived statistics and synthetic-policy checks remain runnable.')
        if args.strict:
            sys.exit(2)


if __name__ == '__main__':
    main()
