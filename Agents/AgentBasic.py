from TrainUtils import *
import torch.nn as nn
import torch.optim as optim

from EnvUtils import VehicleState, VehicleAction
from Models.GreedyModel import GreedyModel

"""
PPO
SAC
"""

class AgentBasic:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, configs: Dict):
        self.configs = configs
        self.model = self.createModel()
        self.model.to(self.device)
        self.model.train()

        self.target_model = self.createModel()
        copyModel(self.model, self.target_model)
        self.target_model.to(self.device)

        self.loss_func = nn.SmoothL1Loss()
        self.optimizer_value = optim.Adam(self.model.getValueParams(), lr=configs["learning_rate"])
        self.optimizer_policy = optim.Adam(self.model.getPolicyParams(), lr=configs["learning_rate"] * 0.1)

        self.gamma = configs["gamma"]

        self.it = 0
        self.target_update_freq = configs["target_model_update_freq"]


    def createModel(self) -> nn.Module:
        return nn.Linear(10, 3)


    def updateTargetModel(self):
        copyModel(self.model, self.target_model)


    def trainStep(self, batch_tensors):
        return 0
        self.optimizer.zero_grad()
        bgr_frame, radar_points, target_dxdydz, action, reward, \
            next_bgr_frame, next_radar_points, next_target_dxdydz, done = batch_tensors

        # TD target = r + gamma * max_a Q(s', a) with target network
        # TD prediction = Q(s, a) with main network

        # Here action is a 3D vector (throttle 0~1, brake 0~1, steer -1~1)
        # Meaning the action space is continuous 3D space
        # So TD target and TD prediction are both 3D vectors
        with torch.no_grad():
            td_target = reward + self.gamma * self.target_model(next_bgr_frame, next_radar_points, next_target_dxdydz) * ~done

        td_prediction = self.student_model()

        loss = self.base_loss_func(td_prediction, td_target)
        loss.backward()
        self.optimizer.step()

        return loss.item()


class OnlyInferAgentBasic:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, base_agent: AgentBasic):
        self.model = base_agent.createModel()
        self.model.to(self.device)
        self.model.eval()

        self.greedy_model = GreedyModel(base_agent.configs)

        self.epsilon = base_agent.configs["epsilon"]

        self.rand_speed = np.random.rand() * 4 - 2
        self.rand_steer = np.random.rand() * 2 - 1
        self.random_count = 0
        self.repeat_mean = base_agent.configs["n_repeat_rand_actions"]
        self.repeat_std = 1.0
        self.repeat_number = self.repeat_mean
        self.greedy_prob = base_agent.configs["greedy_rand_action_prob"]


    def getGreedyAction(self, state) -> VehicleAction:
        lidar_map, spacial_features = state.getTensor()
        V_s, speed_mu, steer_mu = self.greedy_model(lidar_map, spacial_features)
        return VehicleAction(speed_mu, steer_mu)  # (B, 2)


    def updateRandomAction(self) -> None:
        """ Update random action """
        self.repeat_number = int(np.random.normal(self.repeat_mean, self.repeat_std))
        self.rand_speed = np.random.rand() * 4 - 2
        self.rand_steer = np.random.rand() * 2 - 1


    def getAction(self, state: VehicleState) -> VehicleAction:
        """ Just get a random action """
        self.updateRandomAction()
        return VehicleAction(self.rand_speed, self.rand_steer)


