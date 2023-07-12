from typing import *
import numpy as np
import torch
from PathConfigs import *

import carla

class VehicleState:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, rgb_camera: np.ndarray, point_cloud: np.ndarray, gnss_xyz: np.ndarray,
                 next_point_xyz: np.ndarray, imu_arr: np.ndarray):
        self.bgr_camera = rgb_camera.copy()
        self.point_cloud = point_cloud.copy()
        self.gnss_xyz = gnss_xyz.copy()
        self.next_point_xyz = next_point_xyz.copy()
        self.imu_arr = imu_arr.copy()

    def __repr__(self):
        H, W, C = self.bgr_camera.shape
        num_p, dim_p = self.point_cloud.shape
        x, y, z = self.gnss_xyz
        tx, ty, tz = self.next_point_xyz
        return f"VehicleState({H}x{W}x{C} Camera, {num_p}x{dim_p} PointCloud, GNSS = ({x},{y},{z}), Destination = ({tx},{ty},{tz})"

    def __str__(self):
        return self.__repr__()

    def getTensor(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return: (bgr_frame: 1x3xHxW, radar_points: 1xNx4, spacial_features: 1x10)
        spacial_features: dx, dy, dz, accelerometer x, y, z, gyroscope x, y, z, compass
        """
        return torch.tensor(self.bgr_camera, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0) / 255, \
            torch.tensor(self.point_cloud, dtype=torch.float32, device=self.device).unsqueeze(0), \
            torch.cat([torch.tensor(self.next_point_xyz - self.gnss_xyz, dtype=torch.float32, device=self.device).unsqueeze(0),
                          torch.tensor(self.imu_arr, dtype=torch.float32, device=self.device).unsqueeze(0)], dim=1)

    @classmethod
    def makeBatch(cls, states: List["VehicleState"]) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Just make a list of states into batch data
        :param states: a list of VehicleState
        :return: (bgr_frame: Bx3xHxW, points: (Bx400x4), target_dxdydz: Bx3)
        """
        bgr_frames = []
        points = []
        target_dxdydz = []
        for state in states:
            bgr_frame, lidar_points, target_dxdydz_ = state.getTensor()
            bgr_frames.append(bgr_frame)
            points.append(lidar_points)
            target_dxdydz.append(target_dxdydz_)

        # Note that each points can have different number of points
        # So we cannot use torch.cat, instead, we use a list of tensors
        return torch.cat(bgr_frames), torch.cat(points), torch.cat(target_dxdydz)



class VehicleAction:
    """
    steer: [-1, 1]
    throttle_brake: [-2, 2] = [-2, -1] U [-1, 0] U [0, 1] U [1, 2]
    [-2, -1]: backward throttle
    [-1, 0]: brake
    [0, 1]: brake
    [1, 2]: forward throttle
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, throttle_brake: Union[float, torch.Tensor], steer: Union[float, torch.Tensor]):
        if isinstance(throttle_brake, torch.Tensor):
            throttle_brake = throttle_brake.flatten(0).item()
        if isinstance(steer, torch.Tensor):
            steer = steer.flatten(0).item()
        self.throttle_brake = throttle_brake
        self.steer = steer

    def getTensor(self) -> torch.Tensor:
        return torch.tensor([self.throttle_brake, self.steer], dtype=torch.float, device=self.device).unsqueeze(0)

    def __repr__(self):
        return f"VehicleAction(throttle_break={self.throttle_brake:.4f}, steer={self.steer:.4f})"

    def __str__(self):
        return self.__repr__()

    def applyTo(self, vehicle) -> None:

        if self.throttle_brake >= 0:
            reverse = False
            throttle_brake = self.throttle_brake
        else:
            reverse = True
            throttle_brake = -self.throttle_brake

        if throttle_brake >= 1:
            throttle = throttle_brake - 1
            brake = 0
        else:
            throttle = 0
            brake = 1 - throttle_brake

        vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=self.steer, reverse=reverse))


    @staticmethod
    def getIdleAction():
        # Stop moving (full brake) and stop steering
        return VehicleAction(0, 0)

    @staticmethod
    def getNoneAction():
        # Stop moving (no brake and let the car go) and stop steering
        return VehicleAction(1, 0)

    @classmethod
    def makeBatch(cls, actions: List["VehicleAction"]) -> torch.Tensor:
        """
        Just make a list of actions into batch data
        :param actions: a list of VehicleAction
        :return: Bx2
        """
        return torch.cat([action.getTensor() for action in actions])