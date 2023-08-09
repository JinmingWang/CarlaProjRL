import torch
import random
from EnvUtils import *
from Models.PathModel import PathModel
from Agents.HumanAgent import HumanAgent

class PathAgent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, configs: Dict):
        self.configs = configs
        self.model = PathModel(configs)
        self.human_agent = HumanAgent()

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

        human_action, control_signal_received, key_pressed = self.human_agent.getAction()
        if control_signal_received:
            self.random_count = 0  # human control will break the random action sequence
            return human_action

        lidar_map, spacial_features = state.getTensor()
        speed_mu, steer_mu = self.model(lidar_map, spacial_features)
        action = VehicleAction(speed_mu, steer_mu)  # (B, 2)
        return action