from TrainUtils import *
from Agents.AgentBasic import AgentBasic, OnlyInferAgentBasic
import torch.nn as nn
from Models.LidarModel import LidarModelSmall   # this is NN without local path planner
from Agents.HumanAgent import HumanAgent
from Models.RLPathModel import RLPathModel
import cv2


class A2CAgent(AgentBasic):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, configs):
        super().__init__(configs)

    def getPolicyMap(self, mu_x, std_x, mu_y, std_y):
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
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * std_x ** 2 + 2 * std_y ** 2))


    def drawSample(self, mu_x, std_x, mu_y, std_y, lidar_map, V_s) -> None:
        """
        Visualize output of the model
        :param mu_x: mean steer
        :param std_x: std steer
        :param mu_y: mean speed
        :param std_y: std speed
        :param lidar_map: the input lidar map
        :param V_s: the state value
        :return: None
        """
        # localmap: (12, 63, 63)
        policy_map = cv2.cvtColor(self.getPolicyMap(mu_x, std_x, mu_y, std_y), cv2.COLOR_GRAY2BGR)
        lidar_map = 0.125 * lidar_map[0:3, ...] + 0.125 * lidar_map[3:6, ...] + 0.25 * lidar_map[6:9, ...] + 0.5 * lidar_map[9:12, ...]
        lidar_map = cv2.resize(lidar_map.permute(1, 2, 0).detach().cpu().numpy(), dsize=(256, 256))

        temp = np.hstack([policy_map, lidar_map])
        temp[:, 256, :] = 1
        cv2.putText(temp, f"V(s)={V_s:.4f}", (32, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        cv2.imshow("Sample", temp)


    def createModel(self) -> nn.Module:
        """
        Build model here, the model path for Local Path Planner is also deinfed here
        :return: loaded overall model
        """
        model = RLPathModel("Checkpoints/PathFinder/20230815-051714/440000.pth", 10)
        if self.configs.get("model_path") is not None:
            model_path = self.configs["model_path"]
            loadModel(model, model_path)
            print("Model loaded from", model_path)
        return model


    def trainStep(self, batch_tensors: List[torch.Tensor]) -> Tuple[float, float, float, float, str]:
        """
        Perform one training update
        :param batch_tensors: A batch of data
        :return: total_loss, policy_loss, value_loss, imitation_loss, message
        """

        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        batch_S_lidarmap, batch_S_features, batch_A, batch_R, \
            batch_nS_lidarmap, batch_nS_features, batch_T, batch_human_action_mask = batch_tensors

        # batch_A: (B, 2)
        # batch_S_features: (B, 5)

        # Check if NaN in batch_S_features
        if torch.any(torch.isnan(batch_S_features)) or torch.any(torch.isnan(batch_nS_features)):
            print("NaN in batch_S_features or batch_nS_features")
            return 0, 0

        with torch.no_grad():
            # V_sp: (B)
            V_sp, _, _, _, _ = self.target_model(batch_nS_lidarmap, batch_nS_features)
            td_target = batch_R + self.gamma * V_sp * ~batch_T  # (B)

        V_s, speed_mu, speed_std, steer_mu, steer_std = self.model(batch_S_lidarmap, batch_S_features)
        speed_distribution = torch.distributions.Normal(speed_mu, speed_std)
        steer_distribution = torch.distributions.Normal(steer_mu, steer_std)

        advantage = (td_target - V_s).detach()  # (B)
        pi_speed = speed_distribution.log_prob(batch_A[:, 0])
        pi_steer = steer_distribution.log_prob(batch_A[:, 1])
        entropy_speed = speed_distribution.entropy()
        entropy_steer = steer_distribution.entropy()

        # Apply selective loss, if the action is made by human, then do not use policy loss
        policy_loss = 0.1 * (-pi_speed * advantage - 1e-2 * entropy_speed)[~batch_human_action_mask].mean() + \
                      0.1 * (-pi_steer * advantage - 1e-2 * entropy_steer)[~batch_human_action_mask].mean()

        value_loss = self.loss_func(V_s, td_target)  # (B)

        if torch.any(batch_human_action_mask):
            trust_ratio = 1     # This is a factor of how much we trust the human actions
            imitation_loss = trust_ratio * \
                         (self.loss_func(steer_mu[batch_human_action_mask], batch_A[batch_human_action_mask, 1]) + \
                          self.loss_func(speed_mu[batch_human_action_mask], batch_A[batch_human_action_mask, 0]))
        else:
            imitation_loss = torch.zeros(1, device=self.device, dtype=torch.float32)

        loss = policy_loss + value_loss + imitation_loss

        if torch.any(torch.isnan(loss)):
            message = f"NAN LOSS, backward and optim disabled, td_err={value_loss.mean().item():.5f}, policy_loss={policy_loss.mean().item():.5f}"
        else:
            loss.backward()
            self.optimizer_value.step()
            self.optimizer_policy.step()

        if self.it % self.configs["print_freq"] == 0:
            message = f"value_loss={value_loss.mean().item():.4f}, policy_loss={policy_loss.mean().item():.4f}, " \
                f"imitation_loss={imitation_loss.mean().item():.4f}, " \
                f"out_a=({speed_mu[0].item():.4f}:{speed_std[0].item():.4f}, {steer_mu[0].item():.4f}:{steer_std[0].item():.4f}), " \
                f"target_a=({batch_A[0, 0].item():.4f}, {batch_A[0, 1].item():.4f})"
        else:
            message = ""

        # if self.it % self.configs["show_freq"] == 0:
        #     self.drawSample(steer_mu[0].item(), steer_std[0].item(), speed_mu[0].item(), speed_std[0].item(),
        #                     batch_S_lidarmap[0], V_s[0].item())
        #     cv2.waitKey(1)

        self.it += 1
        if self.it % self.target_update_freq == 0:
            copyModel(self.model, self.target_model)

        return loss.item(), policy_loss.item(), value_loss.item(), imitation_loss.item(), message


class OnlyInferA2CAgent(OnlyInferAgentBasic):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, base_agent: A2CAgent):
        super().__init__(base_agent)
        self.human_agent = HumanAgent()


    @torch.no_grad()
    def getActionFromDistribution(self, state: VehicleState) -> VehicleAction:
        """
        This method samples an action from the policy distribution instead of returning the best action
        :param state: input state
        :return: action
        """
        lidar_map, spacial_features = state.getTensor()
        V_s, speed_mu, speed_std, steer_mu, steer_std = self.model(lidar_map, spacial_features)
        # Randomly sample speed from speed distribution
        speed_distribution = torch.distributions.Normal(speed_mu, speed_std)
        speed = speed_distribution.sample()
        steer_distribution = torch.distributions.Normal(steer_mu, steer_std)
        steer = steer_distribution.sample()
        return VehicleAction(speed.item(), steer.item())


    @torch.no_grad()
    def getAction(self, state: VehicleState) -> VehicleAction:
        """
        Get an action from the state (used in simulation, not training)
        :param state: input state
        :return: action
        """

        # If human expert gives control signal, directly use it
        human_action, control_signal_received, key_pressed = self.human_agent.getAction()
        if control_signal_received:
            self.random_count = 0   # human control will break the random action sequence
            return human_action

        # Random Repeat Random Actions
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

        # There is also a percentage to perform greedy action
        # greedily go towards the next waypoint (ignores obstacles)
        if random.random() < self.greedy_prob:
            return self.getGreedyAction(state)
        else:
            lidar_map, spacial_features = state.getTensor()
            V_s, speed_mu, speed_std, steer_mu, steer_std = self.model(lidar_map, spacial_features)
        return VehicleAction(speed_mu, steer_mu)  # (B, 2)

