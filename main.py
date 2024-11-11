from src.environment.environment import Environment
from src.agent.agent import Agent

import numpy as np
import time
import wandb
import threading
import os

wandb.login()
wandb.init(project='tossingbot', name=str(time.time()).split('.')[0])

env = Environment(seed=47)
obs, goal = env.reset()
agent = Agent(robot_base_position=env.robot_base_pos, batch_size=16, run_training=True, device='cuda')

# Metrics
total_correct_grasp = []
total_correct_throw = []

total_steps = 15000
step = 0

accuracy_data = {
    'step': 0,
    'acc_grasp': 0.0,
    'acc_throw': 0.0,
    'explore_prob': agent.explore_prob(step)
}
for key in accuracy_data.keys():
    wandb.define_metric(key, step_metric='step')

def toggle_wait():
    print('Press Enter at any time during training to toggle on/off render speed.')
    while True:
        input()  # Wait for Enter key press
        env.toggle_wait()

toggle_wait_thread = threading.Thread(target=toggle_wait, daemon=True)
toggle_wait_thread.start()


while step < total_steps:

    action = agent.act(obs=obs, goal=goal)
    obs_next, goal_next, reward, info = env.step(action=action)
    if not info['throw_physics_failure']:
        reset_sim = env.check_sim()
        if not reset_sim:
            agent.save_experience(step=step,
                                  action=action, 
                                  observation=obs,
                                  reward=reward, 
                                  goal=goal,
                                  true_landing_pos=info['true_landing_pos'])

            total_correct_grasp.append(reward[0])
            total_correct_throw.append(reward[1])

            if len(total_correct_grasp) > 1000:
                total_correct_grasp = total_correct_grasp[1:]
                total_correct_throw = total_correct_throw[1:]
            accuracy_grasp = np.array(total_correct_grasp).sum()/1000 if len(total_correct_grasp) == 1000 else (np.array(total_correct_grasp).sum()/len(total_correct_grasp)) * (len(total_correct_grasp)/1000)
            accuracy_throw = np.array(total_correct_throw).sum()/1000 if len(total_correct_throw) == 1000 else (np.array(total_correct_throw).sum()/len(total_correct_grasp)) * (len(total_correct_throw)/1000)

            obs = obs_next
            goal = goal_next

            step += 1
            print(f'Step: {step}/{total_steps}: Grasp {("success" if reward[0] else "failure")} (Acc. {accuracy_grasp:.5f}) - Throw {("success" if reward[1] else "failure")} (Acc. {accuracy_throw:.5f})')

            accuracy_data = {
                'step': step,
                'acc_grasp': accuracy_grasp,
                'acc_throw': accuracy_throw,
                'explore_prob': agent.explore_prob(step)
            }
            wandb.log(accuracy_data)

        else:
            print(f'Step: {step}/{total_steps} reset the environment due to unstable sim.')
            obs, goal = env.reset()
    else:
        print(f'Step: {step}/{total_steps} throw physics failure, repeating step.')
        obs = obs_next
        goal = goal_next
        
    time.sleep(0.001)
