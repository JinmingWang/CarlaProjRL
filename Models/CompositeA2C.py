import torch

from Models.ModelUtils import *
from Models.SegModel import FasterSegmentation
from Models.PointCloudModel import PointCloudModel


"""
2023-06-15
Total params: 4,420,337
Total memory: 142.22MB
Total MAdd: 4.47GMAdd
Total Flops: 2.04GFlops
Total MemR+W: 303.97MB
1000 Inference Time on 3070 GPU: 3s - 4s
Comment: Many cameras have fps of 24 to 30, which means 33ms to 41ms per frame, this model is too fast, we can add more
"""
class CompositeA2C(nn.Module):
    """
    Inputs:
        - rgb_cam: (B, 3, H, W)
        - radar_data: (B, Np, 4)
        - target_dxdydz: (B, 3)

    Outputs:
        - Policy: (B, 55, 55) (sum to 1)
        - State Value V(s): (B)


    Architecture:
    - BGR Frame: (B, 3, H, W) -> Frame Encoded (B, 128, 18, 25)
    - Radar Points: (B, Np, 4) -> Radar Encoded (B, 32, 18, 25)
    - Concatenate Frame Encoded and Radar Encoded: (B, 160, 18, 25)
    - Pass through CNN: (B, 160, 18, 25) -> (B, 64, 55, 55)
    - Flatten: (B, 32, 3025)

    - dxdydz linear: (B, 3) -> (B, 96) -> (B, 32, 1, 3)
    - dxdydz multiplied: (B, 32, 3025, 1) * (B, 32, 1, 3) -> (B, 32, 3025, 3)
    - conv3x3 -> (B, 32, 1155, 3) -> (B, 32, 1155, 1)
    - reshape: (B, 32, 1155) -> (B, 32, 21, 55)
    - head: (B, 32, 21, 55) -> (B, 1, 21, 55)
    """
    def __init__(self, configs: Dict):
        super().__init__()

        self.configs = configs

        self.bgr_encoder = self.__createBGREncoder()
        self.points_encoder = self.__createPointsEncoder()

        self.bgr_postprocess = ConvNormAct(256, 128, k=3, s=1, p=1)

        self.img_radar_cat_block = nn.Sequential(
            ConvNormAct(128+32, 256, k=3, s=1, p=1),   # (256, 18, 25)
            ConvNormAct(256, 64, k=3, s=2, p=1),    # (64, 9, 13)
            ConvNormAct(64, 16, k=3, s=2, p=1),     # (16, 5, 7)
            nn.Flatten(start_dim=1),    # (B, 16*5*7=560)
            nn.Linear(560, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),     # (B, 128)
        )

        self.value_head = nn.Sequential(
            nn.Linear(128 + 10, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(128 + 10, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 4),
        )
        # outputs: (B, 5), [V(s), mean(throttle_brake), std(throttle_brake), mean(steer), std(steer)]


    def __createBGREncoder(self, bgr_weight_path: str = None):
            original_model = FasterSegmentation()
            if bgr_weight_path is not None:
                original_model.load_state_dict(torch.load(bgr_weight_path))
            return nn.Sequential(
                original_model.stem,
                original_model.s1,
                original_model.s2,
                original_model.s3,
                original_model.neck,
            )


    def __createPointsEncoder(self):
        return PointCloudModel()


    def getUnfreezeParams(self):
        params = []
        if not self.configs["freeze_bgr_model"]:
            params.append({"params": self.bgr_encoder.parameters()})

        if not self.configs["freeze_point_cloud_model"]:
            params.append({"params": self.points_encoder.parameters()})

        if not self.configs["freeze_connection"]:
            params.append({"params": self.bgr_postprocess.parameters()})
            params.append({"params": self.img_radar_cat_block.parameters()})

        if not self.configs["freeze_head"]:
            params.append({"params": self.policy_head.parameters()})
            params.append({"params": self.value_head.parameters()})

        return params



    def forward(self, rgb_cam, radar_data, spacial_features):
        bgr_enc = self.bgr_postprocess(self.bgr_encoder(rgb_cam))     # (B, 128, 18, 25)
        points_enc = self.points_encoder(radar_data)     # (B, 32, 18, 25)

        img_points_enc = self.img_radar_cat_block(torch.cat([bgr_enc, points_enc], dim=1))     # (B, 128)

        output = self.policy_head(torch.cat([img_points_enc, spacial_features], dim=1))     # (B, 5)

        V_s = self.value_head(torch.cat([img_points_enc, spacial_features], dim=1))     # (B, 1)

        # throttle_brake must have mu in range [-2, 2] and std in range [0, 1]
        throttle_brake_mu = torch.tanh(output[:, 0]) * 2
        throttle_brake_std = torch.sigmoid(output[:, 1])

        # steer must have mu in range [-1, 1] and std in range [0, 1]
        steer_mu = torch.tanh(output[:, 2])
        steer_std = torch.sigmoid(output[:, 3])

        return V_s, throttle_brake_mu, throttle_brake_std, steer_mu, steer_std


def statModel():
    from torchstat import stat
    model = CompositeA2C({})
    replaceLayerByType(model, nn.LeakyReLU, nn.ReLU, inplace=True)

    model_wrapper = ModelWrapper(model, [(3, 600, 800), (400, 4), (10,)])
    stat(model_wrapper, (3, 600, 800))


if __name__ == '__main__':

    # from torchstat import stat
    #
    model = CompositeA2C({})

    # statModel()

    inferSpeedTest(model.cuda(), [(3, 600, 800), (400, 4), (10,)])

