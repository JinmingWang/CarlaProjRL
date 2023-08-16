import torch
from Models.ModelUtils import *
import random
import numpy as np
import cv2
from itertools import product


class OutputHead(nn.Module):
    def __init__(self, out_size: int):
        # input size: (B, 256, 8, 8)
        super().__init__()

        self.s1 = nn.Sequential(
            ConvNormAct(256, 64, 3, 1, 1),  # (256, 8, 8) -> (64, 8, 8)
            ConvNormAct(64, 16, 3, 1, 1),  # (64, 8, 8) -> (16, 8, 8)
            nn.Flatten(1),  # (B, 1024)
            nn.Linear(1024, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
        )

        self.s2 = nn.Sequential(
            nn.Linear(64+5, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, out_size)
        )

    def forward(self, shared_feature, spatial_feature):
        x = self.s1(shared_feature)
        return self.s2(torch.cat([x, spatial_feature], dim=1))


class LocalPathPlanner(nn.Module):
    # Just output action values
    def __init__(self):
        super().__init__()

        # input size: (B, 3, 21, 21)
        self.shared = nn.Sequential(
            ConvNormAct(3, 32, k=3, s=1, p=0),  # (B, 64, 19, 19)
            ConvNormAct(32, 64, k=4, s=1, p=0),  # (B, 64, 16, 16)
            FasterNetBlock(64),
            FasterNetBlock(64),
            ConvNormAct(64, 128, k=3, s=2, p=1),  # (B, 128, 8, 8)
            FasterNetBlock(128),
            FasterNetBlock(128),
            ConvNormAct(128, 256, k=3, s=2, p=1),  # (B, 256, 4, 4)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0),  # (B, 512, 1, 1)
            nn.LeakyReLU(inplace=True),
            nn.Flatten()
        )

        self.actor = nn.Sequential(
            nn.Linear(512, 256), nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128), nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64), nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32), nn.LeakyReLU(inplace=True),
            nn.Linear(32, 5),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Flatten(0)
        )

    def forward(self, grid_world):
        shared_vector = self.shared(grid_world)
        policies = torch.distributions.Categorical(self.actor(shared_vector))
        values = self.critic(shared_vector)
        return policies, values


    def getAction(self, grid_world: torch.Tensor, random_prob: float, greedy_prob: float=0):
        rand = random.random()
        if rand < random_prob:
            # random action
            return random.randint(0, 4)     # 0: up, 1: right, 2: down, 3: left, 4: idle
        elif rand < random_prob + greedy_prob:
            # Greedy select the action towards the target
            size = grid_world.shape[-1]
            current_r, current_c = torch.where(grid_world[0, 1] == 1)
            # best_neighbor = [current_r, current_c]
            best_value = grid_world[0, 0, current_r, current_c]
            best_action = 4
            for action, dr, dc in [(0, -1, 0), (1, 0, 1), (2, 1, 0), (3, 0, -1)]:
                neighbor_r = current_r + dr
                neighbor_c = current_c + dc
                if 0 <= neighbor_r < size and 0 <= neighbor_c < size:
                    value = grid_world[0, 0, neighbor_r, neighbor_c]
                    if value > best_value:
                        best_value = value
                        # best_neighbor = [neighbor_r, neighbor_c]
                        best_action = action
            return best_action
        else:
            # Select follow the model
            policy = self.actor(self.shared(grid_world))     # 5
            action = int(torch.argmax(policy[0]))
            return action


class RLPathModel(nn.Module):
    """
    Inputs:
        - lidar_map: (B, 12, 127, 127) [12 channels is composed of 4 BGR images]
            this lidar map, and the previous 3 lidar maps
        - spacial_features: (B, 5) [x, y, z, compass, speed]

    Outputs:
        - Policy (speed_mu, speed_std, steer_mu, steer_std)
        - State Value V(s): (B)
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    def __init__(self, decision_net_weight: str, n_path_steps: int=10, debug: bool=False):
        super().__init__()
        # this part si already trained
        # During, training, this is applied multiple times just to output the path
        self.path_steps = n_path_steps
        self.path_model = LocalPathPlanner()
        self.path_model.load_state_dict(torch.load(decision_net_weight))
        self.path_model.eval()
        self.debug = debug

        self.body = nn.Sequential(
            ConvNormAct(6, 64, k=3, s=2, p=1),  # (3, 64, 64) -> (64, 32, 32)
            FasterNetBlock(64),
            FasterNetBlock(64),

            ConvNormAct(64, 128, k=3, s=2, p=1),  # (64, 32, 32) -> (128, 16, 16)
            FasterNetBlock(128),
            FasterNetBlock(128),

            ConvNormAct(128, 256, k=3, s=2, p=1),  # (128, 16, 16) -> (256, 8, 8)
            FasterNetBlock(256),
            FasterNetBlock(256)
        )

        self.value_head = OutputHead(out_size=1)
        self.policy_head = OutputHead(out_size=4)

        self._initStepFilter()


    def eval(self):
        self.body.eval()
        self.value_head.eval()
        self.policy_head.eval()
        return self


    def train(self, mode=True):
        self.body.train()
        self.value_head.train()
        self.policy_head.train()
        return self


    def _initStepFilter(self):
        """ This function initializes 5 conv layers, by applying conv layer to agent channel.
        agent_channel = grid_world[:, 1, ...].unsqueeze(1),
        the output will be the next state of the grid world environment.
        In this way, stepping can be embedded into neural networks.
        """
        # step_filter weight shape: (1, 1, 3, 3)
        # after applying corresponding weight to the grid_world[:, 1, ...], the result is the next step
        self.step_filters = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to(self.device) for _ in range(5)])

        up_weight = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=self.device)
        up_weight[:, :, 2, 1] = 1
        self.step_filters[0].weight.data = up_weight

        right_weight = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=self.device)
        right_weight[:, :, 1, 0] = 1
        self.step_filters[1].weight.data = right_weight

        down_weight = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=self.device)
        down_weight[:, :, 0, 1] = 1
        self.step_filters[2].weight.data = down_weight

        left_weight = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=self.device)
        left_weight[:, :, 1, 2] = 1
        self.step_filters[3].weight.data = left_weight

        idle_weight = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=self.device)
        idle_weight[:, :, 1, 1] = 1
        self.step_filters[4].weight.data = idle_weight



    def getValueParams(self):
        param_list = []
        param_list.append({"params": self.body.parameters()})
        param_list.append({"params": self.value_head.parameters()})
        return param_list


    def getPolicyParams(self):
        # Policy may have different learning rate, so we separate it from the value params
        param_list = []
        param_list.append({"params": self.policy_head.parameters()})
        return param_list


    def lidarMap2GridWorld(self, lidar_map):
        # lidar map: (B, 3, 127, 127)
        obstacle_map = (lidar_map[:, 2, ...] > 0.4).unsqueeze(1)  # (B, 1, 127, 127)

        danger_map = func.max_pool2d(obstacle_map.to(torch.float32), kernel_size=6, stride=6).squeeze(1)  # (B, 32, 32)

        # get all target locations
        target_locations = torch.argmax(lidar_map[:, 0].flatten(1), dim=1)  # argmax(B, 127*127) -> (B)
        target_rows = target_locations // 127 // 6  # (B)
        target_cols = target_locations % 127 // 6  # (B)

        # Horizontal distance from target point (B, 32)
        h_dist = torch.abs(
            torch.arange(21, dtype=torch.float32, device=lidar_map.device).view(1, -1) - target_cols.view(-1, 1))
        # Vertical distance from target point (B, 32)
        v_dist = torch.abs(
            torch.arange(21, dtype=torch.float32, device=lidar_map.device).view(1, -1) - target_rows.view(-1, 1))
        # dist_map: (B, 32, 32)
        dist_map = torch.sqrt(h_dist[:, None, :] ** 2 + v_dist[:, :, None] ** 2)

        grid_world = torch.zeros((lidar_map.shape[0], 3, 21, 21), dtype=torch.float32, device=lidar_map.device)
        grid_world[:, 0] = 1 - dist_map / torch.max(dist_map.flatten(0), dim=0).values.view(-1, 1, 1)
        grid_world[:, 1, 9, 10] = 1  # agent location
        grid_world[:, 2] = danger_map  # danger map

        return grid_world


    def generatePath(self, lidar_map):
        B = lidar_map.shape[0]
        grid_world_list = [self.lidarMap2GridWorld(lidar_map)]
        for i in range(self.path_steps):
            policies = self.path_model.actor(self.path_model.shared(grid_world_list[-1]))  # (B, 5)
            actions = torch.argmax(policies[:, :-1], dim=1)  # (B)
            # apply step filter to get next step
            next_grid_world = torch.clone(grid_world_list[-1])
            for b in range(B):
                next_grid_world[b:b + 1, 1:2] = self.step_filters[actions[b]](next_grid_world[b:b + 1, 1:2])
            grid_world_list.append(next_grid_world)

        # all target channels
        target_channel = grid_world_list[0][:, 0]   # (B, 32, 32)
        # 11 * (B, 21, 21) -> (11, B, 21, 21) -> (B, 21, 21)
        path_channel = torch.max(torch.stack([grid_world_list[i][:, 1] * ((self.path_steps - i) / self.path_steps) for i in range(self.path_steps + 1)]), dim=0).values
        obstacle_channel = grid_world_list[0][:, 2]   # (B, 21, 21)
        return torch.stack([target_channel, path_channel, obstacle_channel], dim=1)  # (B, 3, 21, 21)


    def forward(self, lidar_map, spatial_features):
        with torch.no_grad():
            grid_world_with_path = self.generatePath(lidar_map)  # (B, 3, 21, 21)
            surrounding_with_path = grid_world_with_path[:, :, 5:16, 5:16].detach()  # (B, 3, 11, 11)
            surrounding_with_path = func.interpolate(surrounding_with_path, size=(62, 62), mode="bilinear")  # (B, 3, 62, 62)
            surrounding_lidar_map = lidar_map[:, :, 32:94, 32:94]  # (B, 3, 62, 62)
            if self.debug:
                temp = np.ones((62, 124, 3), dtype=np.float32)
                temp[:, :62] = surrounding_with_path[0].detach().permute(1, 2, 0).cpu().numpy()
                temp[:, 62:] = surrounding_lidar_map[0].detach().permute(1, 2, 0).cpu().numpy()
                cv2.imshow("grid_world", cv2.resize(temp, (520, 256), interpolation=cv2.INTER_NEAREST))

        surroundings = torch.cat([surrounding_lidar_map, surrounding_with_path], dim=1)  # (B, 6, 62, 62)

        shared_features = self.body(surroundings)  # (B, 256, 8, 8)

        V_s = self.value_head(shared_features, spatial_features).flatten(0)

        speed_steer = self.policy_head(shared_features, spatial_features)

        speed_mu = torch.tanh(speed_steer[:, 0]) * 2
        speed_std = torch.sigmoid(speed_steer[:, 1]) * 0.6 + 0.4

        steer_mu = torch.tanh(speed_steer[:, 2])
        steer_std = torch.sigmoid(speed_steer[:, 3]) * 0.6 + 0.4

        return V_s, speed_mu, speed_std, steer_mu, steer_std



if __name__ == '__main__':
    from torchstat import stat

    RLPathModel.device = torch.device("cpu")
    model = RLPathModel("Checkpoints/PathFinder/210000.pth", 10, debug=True)

    # stat(model.path_model, (3, 21, 21))
    stat(model, (3, 127, 127))

    # dummy_input = torch.zeros(2, 3, 127, 127, device="cuda")
    # dummy_input[0, 0, 100, 100] = 1   # target is at (5, 5)
    # dummy_input[0, 2, [20, 25, 30, 35, 40], [40, 35, 30, 25, 20]] = 1     # obstacles
    #
    # dummy_input[1, 0, 20, 20] = 1   # target is at (120, 120)
    #
    # dummy_input[:, 2, 64, 74] = 1
    #
    # dummy_spatial_feature = torch.randn(2, 5, device="cuda")
    #
    # print(model(dummy_input, dummy_spatial_feature))
    # cv2.waitKey(0)









