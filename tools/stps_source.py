"""Load original STPS definitions without executing its checkpoint-loading benchmark."""
import ast
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'src/code'
SOURCE = CODE / 'scripts/test/eval_unified.py'
CONSTANTS = {'STALL_WINDOW', 'STALL_DIST', 'BASE_ESCAPE_STEPS', 'PROGRESS_DIST',
             'OSC_WINDOW', 'OSC_REVERSAL_THRESH', 'OSC_MIN_STEPS', 'SEEDS', 'SCENARIOS'}
FUNCTIONS = {'make_configs', 'detect_oscillation', 'run_cnntd3', 'run_stps_v2'}


def load_definitions(sim_class=None):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'), filename=str(SOURCE))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            selected.append(node)
        elif isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) and t.id in CONSTANTS for t in node.targets):
            selected.append(node)
    namespace = {'np': np, 'deque': deque, 'SIM': sim_class}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    if not FUNCTIONS.issubset(namespace) or not CONSTANTS.issubset(namespace):
        raise RuntimeError('Original evaluator definitions changed; review the loader.')
    return namespace
