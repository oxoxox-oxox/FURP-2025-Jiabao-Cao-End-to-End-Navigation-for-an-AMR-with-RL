"""ATD3 训练脚本：Attention-TD3，100点雷达，20×20世界，走廊混合训练"""
import torch, numpy as np, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.CNNTD3.CNNTD3_attention import ATD3
from SIM_ENV.sim import SIM
from utils import get_buffer

def main():
    STATE_DIM   = 105   # 100点雷达 + distance + cos + sin + 2action
    ACTION_DIM  = 2
    MAX_ACTION  = 1
    MAX_EPOCHS  = 100
    EP_PER_EPOCH = 70
    MAX_STEPS   = 300
    BATCH_SIZE  = 64
    TRAIN_EVERY = 2
    TRAIN_ITER  = 80
    SAVE_EVERY  = 5

    STANDARD_WORLDS = [
        'worlds/robot_world_atd3.yaml',
    ]
    CORRIDOR_WORLD = 'worlds/corridor_world_20.yaml'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ATD3(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, max_action=MAX_ACTION,
        device=device, save_every=SAVE_EVERY, load_model=False,
        model_name='ATD3',
        save_directory='models/ATD3/checkpoint',
        load_directory='models/ATD3/checkpoint',
    )

    current_world = STANDARD_WORLDS[0]
    sim = SIM(world_file=current_world, disable_plotting=True)
    replay_buffer = get_buffer(model, sim, False, False, 10, TRAIN_ITER, BATCH_SIZE)

    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(0.0, 0.0)

    epoch, episode, steps = 0, 0, 0

    print(f"ATD3 训练开始 | device={device} | state_dim={STATE_DIM}")

    while epoch < MAX_EPOCHS:
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a)
        action = model.get_action(np.array(state), add_noise=True)
        a_in = [(action[0]+1)/4, action[1]]

        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1])

        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a)
        replay_buffer.add(state, action, reward, terminal, next_state)

        if terminal or steps >= MAX_STEPS:
            # 课程调度：epoch>20后20%概率走廊
            corridor_prob = min(0.2, max(0, (epoch-20)/MAX_EPOCHS))
            if epoch >= 20 and random.random() < corridor_prob:
                new_world = CORRIDOR_WORLD
            else:
                new_world = random.choice(STANDARD_WORLDS)

            if new_world != current_world:
                current_world = new_world
                sim = SIM(world_file=current_world, disable_plotting=True)

            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset(random_obstacles=False)
            outcome = 'GOAL' if goal else ('COL' if collision else 'timeout')
            print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Ep {episode+1}/{EP_PER_EPOCH} | "
                  f"{outcome} | {current_world.split('/')[-1]}", flush=True)

            episode += 1
            steps = 0

            if episode % TRAIN_EVERY == 0:
                model.train(replay_buffer, TRAIN_ITER, BATCH_SIZE)
        else:
            steps += 1

        if episode >= EP_PER_EPOCH:
            episode = 0
            epoch += 1
            # eval
            eval_sim = SIM(world_file=STANDARD_WORLDS[0], disable_plotting=True)
            goals, cols = 0, 0
            for _ in range(10):
                sc,di,co,si,cl,gl,ac,rw = eval_sim.reset()
                done, st = False, 0
                while not done and st < 501:
                    s,_ = model.prepare_state(sc,di,co,si,cl,gl,ac)
                    act = model.get_action(np.array(s), False)
                    sc,di,co,si,cl,gl,ac,rw = eval_sim.step((act[0]+1)/4, act[1])
                    st += 1; done = cl or gl
                    if cl: cols += 1
                    if gl: goals += 1
            print(f"\nEpoch {epoch} Eval: SR={goals/10:.0%} CR={cols/10:.0%}\n")
            model.writer.add_scalar('eval/avg_goal', goals/10, epoch)
            model.writer.add_scalar('eval/avg_col',  cols/10,  epoch)
            model.save('ATD3', 'models/ATD3/checkpoint')

if __name__ == '__main__':
    main()
