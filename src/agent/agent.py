from src.agent.physics_based_controller import PhysicsBasedController
from src.agent.models import TossingbotModel
from src.agent.per_memory import PERMemory
import numpy as np
import torch
import torchvision.transforms.functional as F
import threading
import time
import torch.nn as nn

class Agent:

    def __init__(self, robot_base_position, batch_size, run_training, device):
        # Robot's base position wrt world.
        self.robot_base_position = robot_base_position
        self.batch_size = batch_size
        self.current_step = 0

        # Action
        self.obs_num_channels = 3
        self.obs_pad = 25
        self.z_rotation_angles = np.arange(0, 360, 360/16)
        self.physics_based_controller = PhysicsBasedController(base_position=self.robot_base_position, device=device)
        # Model
        self.device = device
        self.model = TossingbotModel(in_channels=self.obs_num_channels).to(device)
        # Memory
        self.memory = PERMemory(root_dir='')
        # Epsilon-greedy exploration-exploitation policy
        self.explore_prob = lambda step: max(0.5 * np.power(0.9990, step), 0.1)
        self.explore_v_bounds = (1.7, 3.6) 

        # Training
        self.run_training = run_training
        self.keep_training = True
        self.is_training = False
        self.training_thread = threading.Thread(target=self.training_loop_thread)
        self.training_thread.daemon = True
        self.training_thread.start()

    def act(self, obs, goal):
        # Exploit
        # 1. If training, wait for model to finish iteration.
        self.keep_training = False
        while self.is_training:
            time.sleep(0.001)    
        # 2. Execute model.
        z_rotation, coord_height, coord_width, velocity = self.exploit(obs=obs, goal=goal)
        # 3. Reactivate training.
        self.keep_training = True

        # Explore 
        # 1. Sample random grasp.
        explore_grasp = np.random.rand() < self.explore_prob(self.current_step)
        if explore_grasp:
            z_rotation = np.random.choice(self.z_rotation_angles)
            h, w, _ = obs.shape
            coord_height = np.random.randint(h)
            coord_width = np.random.randint(w)

        # 2. Sample random throw.
        explore_throw = np.random.rand() < self.explore_prob(self.current_step)
        if explore_throw:
            velocity = np.random.rand() * (self.explore_v_bounds[1] - self.explore_v_bounds[0]) + self.explore_v_bounds[0]

        self.current_step += 1

        action = {
            'z_rotation': z_rotation,
            'coord_height': coord_height,
            'coord_width': coord_width,
            'velocity': velocity,
            'explore_grasp': explore_grasp,
            'explore_throw': explore_throw
        }
        return action
    
    def exploit(self, obs, goal):
        batch_obs = self.encode_obs_to_action_batch(obs=obs).to(self.device)
        batch_goals = torch.tensor([goal] * len(self.z_rotation_angles), device=batch_obs.device, requires_grad=False, dtype=torch.float32)
        with torch.no_grad():
            v_analytical = self.physics_based_controller.compute_throwing_velocity(goals=batch_goals)
            q_grasp, v_residual = self.model(x=batch_obs[:,:self.obs_num_channels], v=v_analytical)    
            z_rotation, coord_height, coord_width, v_residual = self.decode_max_q_action(q_grasp=q_grasp, v_residual=v_residual)
            velocity = v_analytical[0].item() + v_residual.item()
            z_rotation = z_rotation
        return z_rotation, coord_height, coord_width, velocity

    def decode_max_q_action(self, q_grasp, v_residual):
        for idx, zrot in enumerate(self.z_rotation_angles):
            q_grasp[idx] = F.rotate(img=q_grasp[idx], angle=zrot, expand=False, interpolation=F.InterpolationMode.NEAREST)
            v_residual[idx] = F.rotate(img=v_residual[idx], angle=zrot, expand=False, interpolation=F.InterpolationMode.NEAREST)
        q_grasp = q_grasp[:,:, self.obs_pad:-self.obs_pad, self.obs_pad: -self.obs_pad]
        v_residual = v_residual[:,:, self.obs_pad:-self.obs_pad, self.obs_pad: -self.obs_pad]
        # Get argmax among all q values
        argmax_idxs = torch.unravel_index(torch.argmax(q_grasp), shape=q_grasp.shape)
        idx_rotation = argmax_idxs[0].item()
        coord_height = argmax_idxs[2].item() 
        coord_width = argmax_idxs[3].item()
        v_residual = v_residual[idx_rotation, 0, coord_height, coord_width]
        z_rotation = self.z_rotation_angles[idx_rotation]
        return z_rotation, coord_height, coord_width, v_residual

    def encode_obs_to_action_batch(self, obs):
        obs = torch.tensor(obs).permute(2,0,1)
        obs = F.pad(obs, self.obs_pad)
        batch = None
        for zrot in self.z_rotation_angles:
            obs_rot = self.encode_obs_to_action(obs, zrot).unsqueeze(0)
            batch = obs_rot if batch is None else torch.cat((batch, obs_rot), dim=0)
        return batch

    def encode_obs_to_action(self, obs, zrot):
        obs_rot = F.rotate(img=obs, angle=-zrot, expand=False, interpolation=F.InterpolationMode.NEAREST)
        return obs_rot

    def save_experience(self, step, action, observation, reward, goal, true_landing_pos):
        H, W, C = observation.shape
        h, w = (action['coord_height'], action['coord_width'])
        experience = torch.zeros((C+1, H, W), requires_grad=False)
        experience[:C] = torch.tensor(observation, requires_grad=False).permute(2,0,1)
        experience[-1, h, w] = 1.0
        experience = F.pad(experience, self.obs_pad)
        experience = self.encode_obs_to_action(obs=experience, zrot=action['z_rotation'])
        encoded_obs = experience[:3]
        encoded_hw = torch.unravel_index(torch.argmax(experience[-1]), shape=experience[-1].shape)
        encoded_hw = [coord.item() for coord in encoded_hw]
        if true_landing_pos is not None:
            goal[0] = true_landing_pos[0]
            goal[1] = true_landing_pos[1]

        self.memory.save_experience(step=step,
                                    encoded_obs=encoded_obs,
                                    encoded_hw=encoded_hw,
                                    goal=goal,
                                    target_velocity=action['velocity'],
                                    reward=reward,
                                    explore_grasp=action['explore_grasp'],
                                    explore_throw=action['explore_throw'])

    def training_loop_thread(self):
        """
        As we show in Sec. VI-E, supervising grasps by the accuracy
        of throws eventually leads to more stable grasps and better
        overall throwing performance. The grasping policy learns to
        favor grasps that lead to successful throws, which is a stronger
        requirement than simple grasp success.
        """
        
        self.is_training = False
        
        # Create the optimizer.
        """
        We train our network f by stochastic gradient descent with
        momentum, using fixed learning rates of 10^-4, momentum of 
        0.9, and weight decay 2^-5.
        """
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=0.0001,
                                    momentum=0.9,
                                    weight_decay=2**(-5))
        
        velocity_criterion = nn.SmoothL1Loss(reduction='none')
        grasp_criterion = nn.BCEWithLogitsLoss(reduction='none')

        while self.run_training:
            
            while self.keep_training:
                self.is_training = True
                self.model.train()

                #1. Load batch
                idxs = self.memory.sample_batch(batch_size=self.batch_size)
                if len(idxs) > 0:
                    obs, goals, v_target, grasp_success, regress_throw, hcoord, wcoord = self.load_batch(idxs)
                    obs = obs.to(self.device)
                    goals = goals.to(self.device)
                    v_target = v_target.to(self.device)
                    grasp_success = grasp_success.to(self.device)
                    regress_throw = regress_throw.to(self.device)
                    
                    optimizer.zero_grad()
                    # 2. Forward thru model
                    v_analytical = self.physics_based_controller.compute_throwing_velocity(goals=goals)
                    q_grasp, v_residual = self.model(x=obs, v=v_analytical)   
                    
                    # 3. Compute loss and backprop.
                    B, C, H, W = q_grasp.shape
                    b_idxs = torch.arange(B).unsqueeze(1).expand(B, C) 
                    c_idxs = torch.arange(C).unsqueeze(0).expand(B, C)
                    h_idxs = hcoord.expand(B, C)
                    w_idxs = wcoord.expand(B, C)
                    q_grasp = q_grasp[b_idxs, c_idxs, h_idxs, w_idxs]
                    v_residual = v_residual[b_idxs, c_idxs, h_idxs, w_idxs]
                    v_pred = v_analytical + v_residual
                    
                    velocity_loss = velocity_criterion(v_pred, v_target)
                    velocity_loss = regress_throw * velocity_loss
                    
                    grasp_loss = grasp_criterion(q_grasp, grasp_success)

                    td_error = grasp_loss + velocity_loss
                    loss = td_error.mean()
                    loss.backward()
                    optimizer.step()

                    # Update memory priorities.
                    self.memory.update_priorities(experience_idxs=idxs, experience_priorities=td_error)

                time.sleep(0.001)
            self.is_training = False
            time.sleep(0.01)                

    def load_batch(self, idxs):
        batch_obs = None
        batch_goals = []
        batch_velocities = []
        batch_grasps = []
        batch_throws = []
        batch_hcoord = []
        batch_wcoord = []
        for idx in idxs:
            obs, data = self.memory.load_experience(idx=idx)
            # Observation
            obs = obs.unsqueeze(0)
            batch_obs = torch.cat((batch_obs, obs)) if batch_obs is not None else obs
            # Goals
            batch_goals.append([data['goal_x'], data['goal_y'], data['goal_z']])
            # Target velocities
            batch_velocities.append([data['target_velocity']])
            # Grasp success
            # batch_grasps.append([data['throw_success'] or (data['grasp_success'] and data['explore_throw'])])
            batch_grasps.append([data['throw_success']])
            # Regress throw
            batch_throws.append([data['grasp_success']])
            # Action coordinates
            batch_hcoord.append([data['action_h']])
            batch_wcoord.append([data['action_w']])

        batch_goals = torch.tensor(batch_goals, requires_grad=False, dtype=torch.float32)
        batch_velocities = torch.tensor(batch_velocities, requires_grad=False, dtype=torch.float32)
        batch_grasps = torch.tensor(batch_grasps, requires_grad=False, dtype=torch.float32)
        batch_throws = torch.tensor(batch_throws, requires_grad=False, dtype=torch.float32)
        batch_hcoord = torch.tensor(batch_hcoord, requires_grad=False, dtype=torch.long)
        batch_wcoord = torch.tensor(batch_wcoord, requires_grad=False, dtype=torch.long)
        return batch_obs, batch_goals, batch_velocities, batch_grasps, batch_throws, batch_hcoord, batch_wcoord



