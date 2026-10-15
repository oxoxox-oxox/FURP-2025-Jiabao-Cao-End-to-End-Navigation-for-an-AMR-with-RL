from collections import deque
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from stps_source import load_definitions
from summarize_results import summarize
import numpy as np


class StubPolicy:
    def __init__(self, action):
        self.action = np.array(action)
        self.calls = 0

    def prepare_state(self, *args):
        return np.zeros(185), False

    def get_action(self, *args):
        self.calls += 1
        return self.action


class TraceSIM:
    outcome = None

    def __init__(self, **kwargs):
        self.env = self
        self.x = 0.
        self.ended = False
        TraceSIM.last = self

    def reset(self, **kwargs):
        return np.ones(180), 10., 1., 0., False, False, [0., 0.], 0.

    def get_robot_state(self):
        return np.array([[self.x], [0.], [0.]])

    def step(self, lin, ang):
        self.x += lin * .1
        return np.ones(180), 10-self.x, 1., 0., self.outcome == 'collision', self.outcome == 'goal', [lin, ang], 0.

    def end(self):
        self.ended = True


class STPSTests(unittest.TestCase):
    def setUp(self):
        TraceSIM.outcome = None
        self.ns = load_definitions(TraceSIM)

    def test_oscillation_and_short_history(self):
        detect = self.ns['detect_oscillation']
        self.assertFalse(detect(deque([(0,0)] * 11)))
        self.assertFalse(detect(deque([(i,0) for i in range(12)])))
        self.assertTrue(detect(deque([(i%2,0) for i in range(12)])))

    def test_seeded_start_protocol(self):
        make = self.ns['make_configs']
        a = make([7.5,5],42)
        self.assertEqual(a,make([7.5,5],42))
        self.assertNotEqual(a,make([7.5,5],123))
        self.assertEqual(len(a),12)
        for state in a:
            self.assertLessEqual(abs(state[0][0]-7.5),.3)
            self.assertLessEqual(abs(state[1][0]-5),.3)

    def test_stall_then_escape_and_return(self):
        main, escape = StubPolicy([-1.,0.]), StubPolicy([1.,0.])
        result = self.ns['run_stps_v2'](main,escape,'unused',None,None,150)
        self.assertEqual(result,'timeout')
        self.assertEqual(escape.calls,120)
        self.assertEqual(main.calls,30)
        self.assertTrue(TraceSIM.last.ended)

    def test_no_progress_escape_timeout(self):
        main, escape = StubPolicy([-1.,0.]), StubPolicy([-1.,1.])
        self.ns['run_stps_v2'](main,escape,'unused',None,None,380)
        self.assertEqual(escape.calls,360)
        self.assertEqual(main.calls,20)

    def test_goal_and_collision_terminate(self):
        for outcome in ('goal','collision'):
            TraceSIM.outcome = outcome
            main,escape=StubPolicy([0.,0.]),StubPolicy([0.,0.])
            self.assertEqual(self.ns['run_stps_v2'](main,escape,'unused',None,None,30),outcome)
            self.assertEqual(main.calls,1)
            self.assertEqual(escape.calls,0)
            self.assertTrue(TraceSIM.last.ended)

    def test_archived_results(self):
        means=summarize(ROOT/'src/results/final/unified_comparison.json')
        self.assertAlmostEqual(means['STPS_v2'],.9375)
        self.assertAlmostEqual(means['CNNTD3_baseline'],.673611111111111)

    def test_invalid_result_rejected(self):
        original=json.loads((ROOT/'src/results/final/unified_comparison.json').read_text())
        for values in ([1.,1.], [0.,0.,2.], [0.,0.,float('nan')]):
            data=copy.deepcopy(original)
            data['STPS_v2']['S1_U_trap']['seeds']=values
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'bad.json';path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): summarize(path)


if __name__ == '__main__':
    unittest.main()
