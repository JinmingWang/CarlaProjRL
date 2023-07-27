import torch

from Models.ModelUtils import *
from torchvision.models import resnet18, resnet34, resnet50


class OutputHead(nn.Sequential):
    def __init__(self, out_size):
        super().__init__(
            nn.Linear(128 + 5, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, out_size)
        )

    def forward(self, x, spacial_features):
        return super().forward(torch.cat([x, spacial_features], dim=1))



class LidarModelRes(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        if model_name == "resnet18":
            # 1.96ms NVIDIA 3070
            # 15.1ms cpu
            self.base_model = resnet18(num_classes=128)
        elif model_name == "resnet34":
            # 2.95ms NVIDIA 3070
            # 16.7ms cpu
            self.base_model = resnet34(num_classes=128)
        elif model_name == "resnet50":
            # 3.43ms NVIDIA 3070
            # 18ms cpu
            self.base_model = resnet50(num_classes=128)

        self.base_model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.value_head = OutputHead(out_size=1)
        self.speed_steer_head = OutputHead(out_size=4)

    def forward(self, x, spacial_features):
        x = self.base_model(x)

        V_s = self.value_head(x, spacial_features).flatten(0)
        speed_steer = self.speed_steer_head(x, spacial_features)

        speed_mu = torch.tanh(speed_steer[:, 0]) * 2
        speed_std = torch.sigmoid(speed_steer[:, 1]) * 1.6 + 0.4

        steer_mu = torch.tanh(speed_steer[:, 2])
        steer_std = torch.sigmoid(speed_steer[:, 3]) * 0.6 + 0.4

        if self.training:
            return V_s, speed_mu, speed_std, steer_mu, steer_std
        else:
            return V_s, speed_mu, steer_mu


if __name__ == '__main__':
    inferSpeedTest(LidarModelRes("resnet34"), [(12, 63, 63), (5,)], device="cpu", batch_size=1)


