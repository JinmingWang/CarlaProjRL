import torch

from Models.ModelUtils import *


class OutputHead(nn.Module):
    def __init__(self, out_size: int):
        # input size: (B, 512, 4, 4)
        super().__init__()

        self.convs = nn.Sequential(
            ConvNormAct(512, 128, 3, 2, 1),  # (256, 2, 2)
            nn.Conv2d(128, 128, 2, 1, 0),  # (128, 1, 1)
            nn.ReLU(inplace=True),
            nn.Flatten(1),  # (B, 128)
        )

        self.fc = nn.Sequential(
            nn.Linear(128+5, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, out_size)
        )

    def forward(self, x, spacial_features):
        x = self.convs(x)
        # spatial_features: (B, 5) [dx, dy, dz, compass, speed]
        # dx, dy, dz are distance to the next point in the path, they can take any value, the vehicle can be far from the target
        # compass is in range [0, 1]
        # speed is m/s and can only take positive values, in practice, this should be less than 28 m/s
        # Just for stability, we divide dx, dy, dz by 10
        spacial_features[:, [0, 1, 2, 4]] /= 10.0
        return self.fc(torch.cat([x, spacial_features], dim=1))

    
# With DWConv and FasterNetBlock = 39 s = 25.62 it/s
# Without DWConv = [01:17<00:00, 12.97it/s]
class LidarModelSmall(nn.Module):
    """
    Inputs:
        - lidar_map: (B, 12, 127, 127) [12 channels is composed of 4 BGR images]
            this lidar map, and the previous 3 lidar maps
        - spacial_features: (B, 5) [x, y, z, compass, speed]

    Outputs:
        - Policy (speed_mu, speed_std, steer_mu, steer_std)
        - State Value V(s): (B)
    """
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            ConvNormAct(3, 32, k=5, s=2, p=2),  # (12, 127, 127) -> (32, 64, 64), rf+=5

            FasterNetBlock(32),     # rf+=8
            ConvNormAct(32, 64, k=3, s=2, p=1),  # -> (64, 32, 32), rf=21+=4

            FasterNetBlock(64),     # rf=25+8=33
            FasterNetBlock(64),     # rf=33+8=41
            ConvNormAct(64, 128, k=3, s=2, p=1),  # -> (128, 16, 16), rf=41+8=49

            FasterNetBlock(128),    # rf=49+16=65
            FasterNetBlock(128),    # rf=65+16=81
            ConvNormAct(128, 256, k=3, s=2, p=1),  # -> (256, 8, 8), rf=81+16=97

            FasterNetBlock(256),    # rf=97+32=129
            FasterNetBlock(256),    # rf=129+32=161
            ConvNormAct(256, 512, k=3, s=2, p=1),  # -> (512, 4, 4), rf=161+32=193
        )

        self.value_head = OutputHead(out_size=1)
        self.speed_steer_head = OutputHead(out_size=4)


    def getValueParams(self):
        param_list = []
        param_list.append({"params": self.body.parameters()})
        param_list.append({"params": self.value_head.parameters()})
        return param_list

    def getPolicyParams(self):
        # Policy may have different learning rate, so we separate it from the value params
        param_list = []
        param_list.append({"params": self.speed_steer_head.parameters()})
        return param_list

    @ staticmethod
    def getSteerFromHeatmaps(heatmaps, get_adversarial=False):
        """
        This function takes heatmaps and compute a target steer control
        :param heatmaps: (B, 1, 16, 16) heatmaps
        :return: steer: (B)
        """
        H = heatmaps.shape[-2]
        W = heatmaps.shape[-1]
        if get_adversarial:
            yx_indices = torch.argmin(heatmaps.flatten(1), dim=1)
        else:
            yx_indices = torch.argmax(heatmaps.flatten(1), dim=1)  # B
        relative_dy = yx_indices // W - (H-1) / 2
        relative_dx = yx_indices % W - (W-1) / 2
        steer_rad = torch.atan2(relative_dx, - relative_dy)
        steer_angle = steer_rad * 180 / torch.pi
        steer = func.hardtanh(steer_angle / 70, -1, 1)
        return steer.detach()


    def forward(self, lidar_map, spacial_features):
        # Get Encoding
        x = self.body(lidar_map)

        # Get state values from heatmap
        V_s = self.value_head(x, spacial_features).flatten(0)
        speed_steer = self.speed_steer_head(x, spacial_features)

        # speed ~ N(speed_mu, speed_std)
        # speed_mu in range [-2, 2]
        # speed_std in range [0.4, 2], why? because if std < 0.4, the peak of this gaussian is too sharp
        speed_mu = torch.tanh(speed_steer[:, 0]) * 2
        # speed_std = torch.sigmoid(speed_steer[:, 1]) * 0.6 + 0.4
        speed_std = func.softplus(speed_steer[:, 1]) * 0.8 + 0.2  # softplus(x) = log(1 + exp(x))

        # steer ~ N(steer_mu, steer_std)
        # steer_mu in range [-1, 1]
        # steer_std in range [0.4, 1]
        steer_mu = torch.tanh(speed_steer[:, 2])
        # steer_std = torch.sigmoid(speed_steer[:, 3]) * 0.6 + 0.4
        steer_std = func.softplus(speed_steer[:, 3]) * 0.8 + 0.2

        return V_s, speed_mu, speed_std, steer_mu, steer_std




def trainSpeedTest():
    model = LidarModelSmall()
    model = model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()

    from tqdm import tqdm

    for _ in tqdm(range(1000)):
        dummy_lidar_map = torch.randn(64, 12, 127, 127).cuda()
        dummy_spatial_feature = torch.randn(64, 5).cuda()
        dummy_target = torch.randn(64, 5).cuda()

        optimizer.zero_grad()

        V_s, speed_mu, speed_std, steer_mu, steer_std = model(dummy_lidar_map, dummy_spatial_feature)
        loss = loss_func(torch.stack([V_s, speed_mu, speed_std, steer_mu, steer_std], dim=1), dummy_target)
        loss.backward()
        optimizer.step()




if __name__ == '__main__':
    # inferSpeedTest(LidarModelSmall(), [(12, 63, 63), (5,)], device="cpu", batch_size=1)
    trainSpeedTest()
    # temp_model = ConvNormAct(32, 64, k=3, s=2, p=1)
    # inferSpeedTest(temp_model, [(32, 63, 63)], device="cuda", batch_size=1)


