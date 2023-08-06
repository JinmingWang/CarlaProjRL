from typing import *
import numpy as np
import torch
from PathConfigs import *

import carla

class VehicleState:
    """
    A class used to represent a state, also provide tensor transform methods and make batch methods
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, local_map: np.ndarray, gnss_xyz: np.ndarray, next_point_xyz: np.ndarray, compass: float, speed_kmh: float):
        self.lidar_map = local_map.copy()
        self.gnss_xyz = gnss_xyz.copy()
        self.next_point_xyz = next_point_xyz.copy()
        self.compass = compass
        self.speed_mps = speed_kmh / 3.6

    def __repr__(self):
        x, y, z = self.gnss_xyz
        tx, ty, tz = self.next_point_xyz
        return f"VehicleState(src=({x},{y},{z}), dst=({tx},{ty},{tz}), compass={self.compass:.2f}, speed={self.speed_mps:.2f}m/s"

    def __str__(self):
        return self.__repr__()

    def getTensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert state data into tensors
        :return: lidar map (CxHxW), spacial features [dx, dy, dz, compass, m/s]
        """
        lidar_map = torch.tensor(self.lidar_map, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0) / 255
        spacial_features = torch.cat([torch.tensor(self.next_point_xyz - self.gnss_xyz), torch.tensor([self.compass, self.speed_mps])])

        return lidar_map, spacial_features.unsqueeze(0).to(dtype=torch.float32, device=self.device)


    @classmethod
    def makeBatch(cls, states: List["VehicleState"]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Just make a list of states into batch data
        :param states: a list of VehicleState
        :return: lidar_maps: (BxCx63x63), spacial_features: (Bx5)
        """
        lidar_maps = []
        spacial_features = []
        for state in states:
            lidar_map, target_dxdydz_ = state.getTensor()
            lidar_maps.append(lidar_map)
            spacial_features.append(target_dxdydz_)

        # Note that each lidar_maps can have different number of lidar_maps
        # So we cannot use torch.cat, instead, we use a list of tensors
        return torch.cat(lidar_maps), torch.cat(spacial_features)



class VehicleAction:
    """
    steer: [-1, 1], -1 corresponds to maximum left turn, 1 corresponds to maximum right turn

    speed: [-2, 2] = [-2, -0.5] U [-0.5, 0] U [0, 0.5] U [0.5, 2]
    [-2, -0.5]: backward throttle   -2 is maximum backward throttle
    [-0.5, 0]:  brake               -0.5 is minimum brake
    [0, 0.5]:   brake               0 is maximum brake
    [0.5, 2]:   forward throttle    2 is maximum forward throttle

    Q: Why design speed like this?
    A: Because I want to make the action space continuous, from most negative to most positive, the speed control should
    follow some continuous and symmetric order:
    SPEED_RANGE     MEANING                         SPEED_CHANGE(MATHEMATICALLY)
    [-2, -0.5]:     Negatively increasing speed     speed *= s for s < -1
    [-0.5, 0]:      Negatively decreasing speed     speed *= s for -1 < s < 0
    0:              0 speed                         speed *= 0
    [0, 0.5]:       Positively decreasing speed     speed *= s for 0 < s < 1
    [0.5, 2]:       Positively increasing speed     speed *= s for s > 1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, speed: Union[float, torch.Tensor], steer: Union[float, torch.Tensor], is_human_action: bool=False):
        if isinstance(speed, torch.Tensor):
            speed = speed.flatten(0).item()
        if isinstance(steer, torch.Tensor):
            steer = steer.flatten(0).item()
        self.speed = speed
        self.steer = steer
        self.is_human_action = is_human_action

    def getTensor(self) -> torch.Tensor:
        return torch.tensor([self.speed, self.steer], dtype=torch.float, device=self.device).unsqueeze(0)

    def __repr__(self):
        return f"VehicleAction(speed={self.speed:.4f}, steer={self.steer:.4f})"

    def __str__(self):
        return self.__repr__()

    def applyTo(self, vehicle) -> None:
        """
        Apply this action to a carla.Vehicle instance
        :param vehicle:  carla.Vehicle instance
        """
        if self.speed >= 0:     # Forward
            reverse = False
            throttle_brake = self.speed
        else:                # Backward
            reverse = True
            throttle_brake = -self.speed

        if throttle_brake >= 0.5:
            throttle = (throttle_brake - 0.5) / 1.5
            brake = 0
        else:
            throttle = 0
            brake = (0.5 - throttle_brake) * 2

        vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=self.steer, reverse=reverse))


    @staticmethod
    def getStopAction():
        # Stop moving (full brake) and stop steering
        return VehicleAction(0, 0)

    @staticmethod
    def getIdleAction():
        # Stop moving (no brake and let the car go) and stop steering
        return VehicleAction(0.5, 0)

    @classmethod
    def makeBatch(cls, actions: List["VehicleAction"]) -> torch.Tensor:
        """
        Just make a list of actions into batch data
        :param actions: a list of VehicleAction
        :return: Bx2
        """
        return torch.cat([action.getTensor() for action in actions])