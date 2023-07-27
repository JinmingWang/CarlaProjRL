import torch
import random
from EnvUtils import *
from Models.GreedyModel import GreedyModel

class GreedyAgent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, configs: Dict):
        self.configs = configs
        self.model = GreedyModel(configs)

        self.epsilon = configs["epsilon"]

        self.rand_speed = np.random.rand() * 4 - 2
        self.rand_steer = np.random.rand() * 2 - 1
        self.random_count = 0
        self.repeat_number = configs["n_repeat_rand_actions"]


    def updateRandomAction(self) -> None:
        """ Update random action """
        self.rand_speed = np.random.rand() * 4 - 2
        self.rand_steer = np.random.rand() * 2 - 1


    @torch.no_grad()
    def getAction(self, state: VehicleState) -> VehicleAction:

        # Otherwise, use epsilon-greedy with neural network control signal or random control signal

        # Only if random_iter == 0 and by some chance, we start to apply a series of random actions
        if self.random_count == 0:
            if random.random() < self.epsilon:
                # Once we enter this block, a pair of new random speed and steer is sampled
                # and (look down --->)
                self.updateRandomAction()
                self.random_count += 1
                return VehicleAction(self.rand_speed, self.rand_steer)

        if self.random_count != 0:
            # (---> look here from above) and this block will be executed for the next several getAction call
            # in conclusion, once we take random action, we take one random action many times
            self.random_count = (self.random_count + 1) % self.repeat_number
            return VehicleAction(self.rand_speed, self.rand_steer)

        lidar_map, spacial_features = state.getTensor()
        V_s, speed_mu, steer_mu = self.model(lidar_map, spacial_features)
        return VehicleAction(speed_mu, steer_mu)  # (B, 2)