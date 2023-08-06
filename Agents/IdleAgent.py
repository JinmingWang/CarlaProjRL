import torch
import random
from EnvUtils import *
from Agents.HumanAgent import HumanAgent

class IdleAgent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, configs: Dict):
        self.configs = configs

        self.human_agent = HumanAgent()


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

        return VehicleAction.getIdleAction()