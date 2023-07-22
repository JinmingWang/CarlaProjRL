import torch

from Models.ModelUtils import *


"""
This agent just greedy follow the next point in the path.
Does not care about collision (Assumes no obstacle)
"""

class GreedyModel(nn.Module):
    """
    Inputs:
        - rgb_cam: (B, 3, H, W)
        - radar_data: (B, Np, 4)
        - target_dxdydz: (B, 3)

    Outputs:
        - Policy: (B, 55, 55) (sum to 1)
        - State Value V(s): (B)
    """
    def __init__(self, configs: Dict):
        super().__init__()

        self.configs = configs

        self.value_head = nn.Sequential(
            nn.Linear(10, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Flatten(0)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(10, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 4),
        )
        # outputs: (B, 5), [V(s), mean(throttle_brake), std(throttle_brake), mean(steer), std(steer)]


    def getUnfreezeParams(self):
        params = []
        params.append({"params": self.policy_head.parameters()})
        params.append({"params": self.value_head.parameters()})

        return params

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
        return greedy_steer.detach()


    def forward(self, rgb_cam, radar_data, spacial_features):

        V_s = self.value_head(spacial_features)

        steer_mu = self.getSteer(spacial_features)
        steer_std = torch.ones_like(steer_mu) * 0.1

        throttle_brake_mu = torch.ones_like(steer_mu) * 1.1
        throttle_brake_std = torch.ones_like(steer_mu) * 0.1

        return V_s, throttle_brake_mu, throttle_brake_std, steer_mu, steer_std

