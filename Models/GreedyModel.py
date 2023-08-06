import torch

from Models.ModelUtils import *


"""
This agent just greedy follow the next point in the path.
Does not care about collision (Assumes no obstacle)
"""

class GreedyModel():
    def __init__(self, configs: Dict):
        super().__init__()

        self.configs = configs

    @staticmethod
    def getSteer(spacial_features):
        # Step 1. Compute steer to let the vehicle directly point to the next point in the path
        absolute_dx = spacial_features[:, 0]  # dx
        absolute_dy = spacial_features[:, 1]  # dy
        theta = - spacial_features[:, -2]  # compass
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        relative_dx = cos * absolute_dx - sin * absolute_dy
        relative_dy = sin * absolute_dx + cos * absolute_dy

        steer_rad = torch.atan2(relative_dx, - relative_dy)
        steer_angle = steer_rad * 180 / torch.pi
        greedy_steer = func.hardtanh(steer_angle / 70, -1, 1)
        return greedy_steer.detach(), -torch.sign(relative_dy)


    def forward(self, lidar_map, spacial_features):
        # spacial_features: (B, 5) [dx, dy, dz, compass, speed]
        steer_mu, throttle_direction = self.getSteer(spacial_features)
        speed_mu = throttle_direction * 1.3
        state_value = torch.zeros_like(steer_mu)
        return state_value, speed_mu, steer_mu

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


