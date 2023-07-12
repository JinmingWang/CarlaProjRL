import torch

from Models.ModelUtils import *

class PointCloudModel(nn.Module):
    def __init__(self):
        super().__init__()

        # input tensor size: (B, P, 4)
        # P = number of points
        # 4 = (altitude (rad), azimuth (rad), distance (meter), velocity (velocity towards the radar))
        # If Lidar
        # Then P = 400
        # 4 = (x, y, z, intensity)

        self.s1 = nn.Sequential(
            nn.Linear(4, 32),   # (1, 400, 4) -> (1, 400, 64)
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(32, 128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(128, 512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # (1, 400, 512)
            nn.Unflatten(dim=1, unflattened_size=(50, -1))  # (1, 400, 512) -> (1, 40, 10, 512)
        )

        self.s2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(1024, 1800),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Unflatten(dim=2, unflattened_size=(4, 18, 25))  # (B, 40, 1800) -> (B, 40, 4, 18, 25)
        )


        self.head = nn.Sequential(
            ConvNormAct(4, 32, 3, 1, 1),
            FasterNetBlock(32)
        )


    def forward(self, x):
        x = torch.mean(self.s1(x), dim=2)  # (B, 40, 512)
        global_feature = torch.mean(self.s2(x), dim=1)  # (B, 4, 18, 25)
        return self.head(global_feature)  # (1, 32, 18, 25)
