from TrainUtils import *
import torch.nn as nn
import torch.optim as optim

from EnvUtils import VehicleState, VehicleAction

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
        self.optimizer = optim.Adam(self.model.getUnfreezeParams(), lr=configs["learning_rate"])

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

        loss = self.loss_func(td_prediction, td_target)
        loss.backward()
        self.optimizer.step()

        return loss.item()


class ShadowAgentBasic:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, base_agent: AgentBasic):
        self.model = base_agent.createModel()
        self.model.to(self.device)
        self.model.eval()

        self.epsilon = base_agent.configs["epsilon"]
        self.epsilon_decay = base_agent.configs["epsilon_decay"]
        self.epsilon_min = base_agent.configs["epsilon_min"]

        self.rand_throttle_brake = np.random.rand() * 4 - 2
        self.rand_steer = np.random.rand() * 2 - 1
        self.random_iter = 0
        self.switch_iter = 10

    def getRandomAction(self) -> VehicleAction:
        # return action as a 3D vector (throttle 0~1, brake 0~1, steer -1~1)
        if self.random_iter < self.switch_iter:
            self.random_iter += 1
        else:
            self.random_iter = 0
            self.rand_throttle_brake = np.random.rand() * 4 - 2
            self.rand_steer = np.random.rand() * 2 - 1

        return VehicleAction(self.rand_throttle_brake, self.rand_steer)


    def getAction(self, state: VehicleState) -> VehicleAction:
        # return action as a 3D vector (throttle 0~1, brake 0~1, steer -1~1)
        return self.getRandomAction()


