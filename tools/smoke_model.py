"""Check original CNNTD3 observation encoding and random-weight actor on CPU."""
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/code'))

import numpy as np
import torch
from robot_nav.models.cnntd3 import CNNTD3
from robot_nav.sim.sim import SIM


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    output = ROOT / 'artifacts'
    output.mkdir(exist_ok=True)
    os.chdir(output)
    sim = SIM(str(ROOT / 'src/code/robot_nav/worlds/narrow_door_world.yaml'), True)
    model = CNNTD3(state_dim=185, action_dim=2, max_action=1,
                   device=torch.device('cpu'), load_model=False, model_name='smoke_random_weights')
    try:
        obs = sim.reset(robot_state=[[2.],[5.],[0.]],robot_goal=[[8.],[5.],[0.]],random_obstacles=False)
        scan,dist,cos,sin,collision,goal,previous,reward = obs
        state, terminal = model.prepare_state(scan,dist,cos,sin,collision,goal,previous)
        assert len(state) == 185 and np.isfinite(state).all() and terminal == 0
        action = model.get_action(np.array(state), False)
        assert action.shape == (2,) and np.isfinite(action).all() and (abs(action) <= 1).all()
        q1,q2 = model.critic(torch.tensor([state],dtype=torch.float32),torch.tensor([action],dtype=torch.float32))
        assert q1.shape == q2.shape == (1,1) and torch.isfinite(q1).all() and torch.isfinite(q2).all()
        obs = sim.step(float(np.clip((action[0]+1)/4,0,.5)),float(np.clip(action[1],-1,1)))
        assert len(obs[0]) == 180 and np.isfinite(obs[1]) and np.isfinite(obs[-1])
        # Verify original normalization semantics and invalid observation rejection.
        state,terminal=model.prepare_state(np.full(180,np.inf),10.,1.,0.,True,False,[.5,0.])
        assert state[:180] == [1.]*180 and state[-5:] == [1.,1.,0.,1.,.5] and terminal == 1
        try:
            model.prepare_state(np.ones(179),1.,1.,0.,False,False,[0.,0.])
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid scan dimension was accepted')
    finally:
        sim.env.end()
        model.writer.close()
    print('MODEL SMOKE PASSED: 180 LiDAR beams -> 185-state encoding -> CPU actor/critics -> simulator step.')
    print('Random initial weights only; no learned navigation performance is asserted.')


if __name__ == '__main__':
    main()
