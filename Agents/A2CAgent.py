from TrainUtils import *
from Agents.AgentBasic import AgentBasic, ShadowAgentBasic
import torch.nn as nn
import torch.optim as optim
import yaml
from Models.CompositeA2C import CompositeA2C
from Models.GreedyModel import GreedyModel
from Agents.HumanAgent import HumanAgent

"""
Just change DQN to A2C
"""


class A2CAgent(AgentBasic):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, configs):
        super().__init__(configs)
        self.policy_map = np.zeros((256, 256), dtype=np.float32)

    def draw2DGaussian(self, mu_x, std_x, mu_y, std_y):
        """
        Draw a 2D gaussian on the policy map
        :param mu_x: x mean [-1, 1]
        :param std_x: x std
        :param mu_y: y mean [-2, 2]
        :param std_y: y std
        :return:
        """
        std_x *= 256
        std_y *= 256
        x = np.arange(0, 256, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = (mu_x * 0.5 + 0.5) * 256
        y0 = 256 - (mu_y + 2) / 4 * 256
        self.policy_map += np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * std_x ** 2 + 2 * std_y ** 2))


    def createModel(self) -> nn.Module:
        model_config_path = self.configs["model_config"]
        with open(model_config_path, "r") as f:
            model_configs = yaml.load(f, Loader=yaml.FullLoader)
        model = CompositeA2C(model_configs)
        if self.configs.get("model_path") is not None:
            model_path = self.configs["model_path"]
            loadModel(model, model_path)
        return model


    def trainStep(self, batch_tensors):

        self.optimizer.zero_grad()
        batch_S_bgr, batch_S_radar, batch_S_features, \
            batch_A, batch_R, \
            batch_nS_bgr, batch_nS_radar, batch_nS_features, \
            batch_T = batch_tensors

        # batch_A: (B, 2)
        # batch_S_features: (B, 10)

        # Check if NaN is in batch_S_features
        if torch.any(torch.isnan(batch_S_features)):
            print(batch_S_features)

        # TD target = r + gamma * V(s') with target network
        # TD prediction = V(s) with main network
        # Advantage = TD target - V(s) with main network
        # policy loss = -log_prob(a) * Advantage
        # value loss = (TD prediction - TD target)^2

        with torch.no_grad():
            # V_sp: (B)
            V_sp, _, _, _, _ = self.target_model(batch_nS_bgr, batch_nS_radar, batch_nS_features)
            td_target = batch_R + self.gamma * V_sp * ~batch_T  # (B)

        # V_s: (B), throttle_brake_mu_std: (B, 2), steer_mu_std: (B, 2)
        V_s, throttle_brake_mu, throttle_brake_std, steer_mu, steer_std = self.model(batch_S_bgr, batch_S_radar, batch_S_features)
        throttle_brake_std = torch.clip(throttle_brake_std, 0.0001, 1)
        steer_std = torch.clip(steer_std, 0.0001, 1)
        throttle_brake_distribution = torch.distributions.Normal(throttle_brake_mu, throttle_brake_std)
        steer_distribution = torch.distributions.Normal(steer_mu, steer_std)

        advantage = (td_target - V_s).detach()  # (B)
        advantage = torch.nn.functional.relu(advantage)
        pi_s_a = torch.clip(throttle_brake_distribution.log_prob(batch_A[:, 0]) + steer_distribution.log_prob(batch_A[:, 1]), -10, 10)  # (B)
        policy_loss = (-pi_s_a * advantage).mean()

        # entropy = throttle_brake_distribution.entropy() + steer_distribution.entropy()

        value_loss = self.loss_func(V_s, td_target)  # (B)

        loss = 0.01 * policy_loss + value_loss# - 0.01 * entropy.mean()

        if torch.any(torch.isnan(loss)):
            print(f"NAN LOSS, backward and optim disabled, td_err={value_loss.mean().item():.5f}, policy_loss={policy_loss.mean().item():.5f}")
        else:
            loss.backward()
            self.optimizer.step()

        if self.it % 100 == 0:
            print(f"value_loss={value_loss.mean().item():.5f}, policy_loss={policy_loss.mean().item():.5f}, "
                  f"out_a=({throttle_brake_mu[0].item():.5f}:{throttle_brake_std[0].item():.5f}, {steer_mu[0].item():.5f}:{steer_std[0].item():.5f}), "
                  f"target_a=({batch_A[0, 0].item():.5f}, {batch_A[0, 1].item():.5f})")
            self.draw2DGaussian(steer_mu[0].item(), steer_std[0].item(), throttle_brake_mu[0].item(), throttle_brake_std[0].item())

        self.it += 1
        if self.it % self.target_update_freq == 0:
            copyModel(self.model, self.target_model)

        return loss.item(), value_loss.mean().item(), policy_loss.mean().item()


class ShadowA2CAgent(ShadowAgentBasic):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, base_agent: A2CAgent):
        super().__init__(base_agent)
        self.human_agent = HumanAgent()

    @torch.no_grad()
    def getAction(self, state: VehicleState) -> VehicleAction:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # If human expert gives control signal, directly use it
        # This makes the algorithm more like imitation learning
        human_action, control_signal_received, key_pressed = self.human_agent.getAction()
        if control_signal_received:
            return human_action

        # Otherwise, use epsilon-greedy with neural network control signal or random control signal
        if random.random() < self.epsilon:
            return self.getRandomAction()
        else:
            bgr_frame, radar_points, target_dxdydx = state.getTensor()
            V_s, throttle_brake_mu, throttle_brake_std, steer_mu, steer_std = self.model(bgr_frame, radar_points, target_dxdydx)
            return VehicleAction(throttle_brake_mu, steer_mu)  # (B, 2)

