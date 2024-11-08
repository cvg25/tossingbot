from PIL import Image
from pathlib import Path
import os
import time
import csv
import numpy as np
import torchvision.transforms as T

class PERMemory():

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.log_path = self.root_dir/'training_data'
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.run_path = self.log_path/str(time.time()).split(".")[0]
        os.mkdir(self.run_path)
        self.experiences_path = self.run_path/'experiences'
        os.mkdir(self.experiences_path)
        self.experiences_file = self.run_path/'experiences.csv'
        self.get_fname_from_step = lambda step: f'{step:06d}.png'
        self.experience_priorities = []
        self.experience_data = []
        self.to_tensor = T.ToTensor()

    def sample_batch(self, batch_size, alpha=0.8):
        """
        Sample batch of idxs from the experience with prioritized experience replay.
        Args:
        - batch_size: total number of samples.
        - alpha: How much prioritization is used (0 - no prioritization, 1 - full prioritization).
        """
        experience_idxs = []
        total_samples = len(self.experience_priorities)
        if total_samples >= batch_size:
            # 1. Sort priorities by temporal difference error. 
            idxs_sorted_by_priorities = np.argsort(self.experience_priorities)[::-1] # Highest priority, first.
            # 2. Compute sampling probabilities using a power-law distribution
            probs = 1.0 / np.power(np.arange(1, total_samples + 1), alpha)
            probs /= probs.sum()
            # 3. Sample indices based on computed probabilities.
            experience_idxs = np.random.choice(idxs_sorted_by_priorities, batch_size, p=probs, replace=False).tolist()
        return experience_idxs
    
    def save_experience(self, step, encoded_obs, encoded_hw, goal, target_velocity, reward, explore_grasp, explore_throw):
        h = encoded_hw[0]
        w = encoded_hw[1]
        experience_dict = {
            'step': step,
            'action_h': h,
            'action_w': w,
            'goal_x': goal[0],
            'goal_y': goal[1],
            'goal_z': 0.0,
            'target_velocity': target_velocity,
            'grasp_success': reward[0],
            'throw_success': reward[1],
            'explore_grasp': explore_grasp,
            'explore_throw': explore_throw
        }
        # 1. Save experience to file.
        self.log_to_file(experience_dict)

        # 2. Save experience img data.
        experience_fpath = self.experiences_path/self.get_fname_from_step(step)
        img = Image.fromarray((encoded_obs * 255).permute(1,2,0).detach().cpu().numpy().astype('uint8'))
        img.save(experience_fpath)

        # 3. Make it available for training.
        self.experience_data.append(experience_dict)
        self.experience_priorities.append(np.inf)

    def load_experience(self, idx):
        assert idx < len(self.experience_data), 'Error: idx is bigger than the number of experiences available.'
        experience_data = self.experience_data[idx]
        experience_fpath = self.experiences_path/self.get_fname_from_step(experience_data['step'])
        obs = Image.open(experience_fpath)
        obs = self.to_tensor(obs)
        return obs, experience_data

    def update_priorities(self, experience_idxs, experience_priorities):
        print(f'Mean TD: {experience_priorities.mean()}')
        for idx, priority in zip(experience_idxs, experience_priorities):
            self.experience_priorities[idx] = priority.item()

    def log_to_file(self, experience_dict):
        write_header = not self.experiences_file.exists()
        with open(self.experiences_file, 'a', newline='') as f:
            csvwriter = csv.writer(f)
            if write_header:
                csvwriter.writerow(list(experience_dict.keys())) 
            csvwriter.writerow(experience_dict.values())