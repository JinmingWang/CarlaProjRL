import torch

from Models.ModelUtils import *
from Models.SegModel import FasterSegmentation
from Models.PointCloudModel import PointCloudModel


"""
This agent just greedy follow the next point in the path.
Does not care about collision (Assumes no obstacle)
"""

class GreedyA2C(nn.Module):
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



    def forward(self, rgb_cam, radar_data, spacial_features):
        output = self.policy_head(spacial_features)     # (B, 5)

        V_s = self.value_head(spacial_features)     # (B, 1)

        # throttle_brake must have mu in range [-2, 2] and std in range [0, 1]
        throttle_brake_mu = torch.tanh(output[:, 0]) * 2
        throttle_brake_std = torch.sigmoid(output[:, 1])

        # steer must have mu in range [-1, 1] and std in range [0, 1]
        steer_mu = torch.tanh(output[:, 2])
        steer_std = torch.sigmoid(output[:, 3])

        return V_s, throttle_brake_mu, throttle_brake_std, steer_mu, steer_std

