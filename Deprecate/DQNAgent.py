from TrainUtils import *
from Agents.AgentBasic import AgentBasic, ShadowAgentBasic
import torch.nn as nn
import yaml
from Deprecate.CompositeDQN import CompositeDQN
from Agents.HumanAgent import HumanAgent

"""
Update design: our action space is (21, 55), totally 1155 actions,
but the actual action space is continuous
which means if two actions are closer to each other, they are more similar
if one action is the best, the actions around it are also good
if one action is the worst, the actions around it are also bad

Therefore, we may not update single action at each time step
The original DQN update is:
Q(s', a'): (21, 55)
max_a': (2) (row index, column index)
TD target = r + gamma * max_a' Q(s', a')
TD prediction = Q(s, a)
loss = (TD target - TD prediction)^2

The new update is:
TD target = r + gamma * max_a' Q(s', a')
Know TD target is a more accurate estimation of the expected return by taking action a at state s
According to our design, other actions near a should expected to have similar return
max_action_loss = (TD target - TD prediction)^2
G = make a (21, 55) gaussian blur centering at action a
action_space_loss = G * max_action_loss

In this way, Q(s, a) still gets updated equal to the original DQN update, but its neighbors also get updated, the update
is weaker as the distance to a increases
"""
class GaussianActionLoss(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, sigma: float=2.0):
        super().__init__()
        self.sigma = sigma
        self.ACTION_X_COORDS = torch.arange(55, dtype=torch.float32, device=self.device).view(1, 1, -1)
        self.ACTION_Y_COORDS = torch.arange(55, dtype=torch.float32, device=self.device).view(1, -1, 1)

    def getGaussianMaps(self, rows: torch.Tensor, cols: torch.Tensor, sigma: float=2.0) -> torch.Tensor:
        """
        Get gaussian map of shape (B, 55, 55)
        result[b] has a 2D gaussian blob centered at (rows[b], cols[b]), the gaussian peaks at 1.0
        :param rows: (B) row indices
        :param cols: (B) column indices
        :param sigma: sigma of the gaussian
        :return: (B, 55, 55) gaussian maps
        """
        # Constants
        B = rows.shape[0]  # Batch size

        # Center coordinates
        rows = rows.view(B, 1, 1)
        cols = cols.view(B, 1, 1)

        # Calculate squared distance from each grid point to the center
        dist_sq = (self.ACTION_X_COORDS - cols) ** 2 + (self.ACTION_Y_COORDS - rows) ** 2

        # Compute Gaussian weights using the squared distances and sigma
        gaussian_weights = torch.exp(-0.5 * dist_sq / (sigma ** 2))
        return gaussian_weights


    def forward(self, Q_s, td_target, batch_A):
        # TD target for (s, a) = r + gamma * max_a' Q(s', a')
        # Original value for (s, a) = Q(s, a)

        # value field for s = Q(s)
        # target field for s = gaussian_map(TD target)

        gaussian_maps = self.getGaussianMaps(batch_A[:, 0], batch_A[:, 1])  # (B, 55, 55)
        update_masks = gaussian_maps >= 0.1
        gaussian_maps *= td_target.unsqueeze(-1).unsqueeze(-1)  # (B, 55, 55)
        return (gaussian_maps[update_masks] - Q_s[update_masks]).pow(2).mean()


class DQNAgent(AgentBasic):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, configs):
        super().__init__(configs)

        self.loss_func = GaussianActionLoss(sigma=2.0)


    def createModel(self):
        model_config_path = self.configs["model_config"]
        with open(model_config_path, "r") as f:
            model_configs = yaml.load(f, Loader=yaml.FullLoader)
        return CompositeDQN(model_configs)


    def trainStep(self, batch_tensors):

        self.optimizer.zero_grad()
        batch_S_bgr, batch_S_radar, batch_S_dxdydz, \
            batch_A, batch_R, \
            batch_nS_bgr, batch_nS_radar, batch_nS_dxdydz, \
            batch_T = batch_tensors

        # batch_A: (B, 2)

        # TD target = r + gamma * max_a Q(s', a) with target network
        # TD prediction = Q(s, a) with main network

        with torch.no_grad():
            # action_values: (B, 55, 55)
            Q_sp_ap = self.target_model(batch_nS_bgr, batch_nS_radar, batch_nS_dxdydz)
            max_ap_Q_sp_ap = Q_sp_ap.view(-1, 55 * 55).max(dim=1)[0]  # (B)
            td_target = batch_R + self.gamma * max_ap_Q_sp_ap * ~batch_T    # (B)

        # Q_s (B, 55, 55)
        Q_s = self.model(batch_S_bgr, batch_S_radar, batch_S_dxdydz)
        # batch_indices = torch.arange(Q_s.shape[0], device=self.device)
        # Q_s_a = Q_s[batch_indices, batch_A[:, 0], batch_A[:, 1]]  # (B)

        loss = self.loss_func(Q_s, td_target, batch_A)

        loss.backward()

        self.optimizer.step()

        self.it += 1
        if self.it % self.target_update_freq == 0:
            copyModel(self.model, self.target_model)

        return loss.item()


class ShadowDQNAgent(ShadowAgentBasic):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, base_agent: DQNAgent):
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
            Q_s = self.model(bgr_frame, radar_points, target_dxdydx)
            flat_argmax = torch.argmax(Q_s.view(-1, 55*55), dim=1)
            throttle_brakes = flat_argmax // 55    # (B)
            steers = flat_argmax % 55  # (B)
            return VehicleAction.fromTensor(torch.stack([throttle_brakes, steers], dim=1))  # (B, 2)


