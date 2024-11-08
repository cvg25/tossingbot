import torch

class PhysicsBasedController:

    def __init__(self, base_position, device):
        self.base_position = base_position
        # Data from the paper.
        self.c_d = 0.7 + torch.linalg.norm(torch.tensor(self.base_position[:2])).to(device) # dist(robot_base_xy, release_xy)
        self.c_h = 0.04 + torch.linalg.norm(torch.tensor(self.base_position[2:])).to(device) # dist(robot_base_z, release_z)
        self.release_angle = torch.deg2rad(torch.tensor([[45]])).to(device)
        self.g = 9.8

    def compute_throwing_velocity(self, goals):
        with torch.no_grad():
            x = torch.linalg.norm(goals[:,0:2], dim=1) - self.c_d
            y = torch.linalg.norm(self.c_h - goals[:, 2:], dim=1)
            v = torch.sqrt((0.5*self.g*x**2)/((y+x*torch.tan(self.release_angle))*0.5)).T # goal throwing release velocity
        return v