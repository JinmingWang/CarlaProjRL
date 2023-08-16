"""
This is the train code for Local Path Planner
"""

import os

from torchvision import transforms
from torch import Tensor

import yaml
from TrainUtils import *
from torch.utils.tensorboard import SummaryWriter

from Models.RLPathModel import LocalPathPlanner
from Models.ModelUtils import *
from torch.utils.data import DataLoader, Dataset
import cv2


RESUME = "Checkpoints/PathFinder/20230815-051714/440000.pth"
RAND_PROB = 0.3
GREEDY_PROB = 0.1
GAMMA = 0.9
MAX_STEPS = 80
BATCH_SIZE = 256
CRITIC_LR = 1e-3
ACTOR_LR = 1e-4


class GridWorldDataset(Dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    dataset directory structure:
    YYYYMMDD-HHMMSS
        - 1
            - 00000.pt
            - 00001.pt
            - ...
        - 2
        - ...

    """
    def __init__(self, memory_dirs: List[str]):
        self.memory_dirs = memory_dirs

        self.file_paths = []
        for memory_dir in memory_dirs:
            for checkpoint_dir in os.listdir(memory_dir):
                checkpoint_path = os.path.join(memory_dir, checkpoint_dir)
                for memory_file in os.listdir(checkpoint_path):
                    memory_path = os.path.join(checkpoint_path, memory_file)
                    self.file_paths.append(memory_path)

        print(f"MemoryDataset: {len(self.file_paths)} files found.")

        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Lambda(self.getRandomObstacleTransform(0.01))
        ])

    def getRandomObstacleTransform(self, prob: float) -> Callable[[torch.Tensor], torch.Tensor]:
        def randomPlaceObstacle(x: torch.Tensor):
            B, C, H, W = x.shape
            obstacle_mask = torch.rand(B, H, W, device=x.device) < prob
            x[:, 2, :, :] = torch.max(x[:, 2, :, :], obstacle_mask.to(torch.float32))
            return x
        return randomPlaceObstacle


    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def lidarMap2GridWorld(lidar_map):
        # lidar map: (B, 3, 127, 127)
        obstacle_map = (lidar_map[:, 2, ...] > 0.4).unsqueeze(1)  # (B, 1, 127, 127)

        danger_map = func.max_pool2d(obstacle_map.to(torch.float32), kernel_size=6, stride=6).squeeze(1)  # (B, 21, 21)

        # get all target locations
        target_locations = torch.argmax(lidar_map[:, 0].flatten(1), dim=1)  # argmax(B, 127*127) -> (B)
        target_rows = target_locations // 127 // 6  # (B)
        target_cols = target_locations % 127 // 6  # (B)

        # Horizontal distance from target point (B, 21)
        h_dist = torch.abs(torch.arange(21, dtype=torch.float32, device=lidar_map.device).view(1, -1) - target_cols.view(-1, 1))
        # Vertical distance from target point (B, 21)
        v_dist = torch.abs(torch.arange(21, dtype=torch.float32, device=lidar_map.device).view(1, -1) - target_rows.view(-1, 1))
        # dist_map: (B, 21, 21)
        dist_map = torch.sqrt(h_dist[:, None, :] ** 2 + v_dist[:, :, None] ** 2)

        grid_world = torch.zeros((lidar_map.shape[0], 3, 21, 21), dtype=torch.float32, device=lidar_map.device)
        grid_world[:, 0] = 1 - dist_map / torch.max(dist_map.flatten(0), dim=0).values.view(-1, 1, 1)
        grid_world[:, 1, 9, 10] = 1  # agent location
        grid_world[:, 2] = danger_map  # danger map

        return grid_world


    def getSource(self, idx):
        state: VehicleState = torch.load(self.file_paths[idx])[0]

        lidar_map, _ = state.getTensor()  # lidar_map: (B, 3, 127, 127)
        return lidar_map


    def __getitem__(self, idx) -> torch.Tensor:
        state: VehicleState = torch.load(self.file_paths[idx])[0]

        lidar_map, _ = state.getTensor()    # lidar_map: (B, 3, 127, 127)
        grid_world = self.lidarMap2GridWorld(lidar_map)     # grid_world: (B, 3, 32, 32)
        # return grid_world
        return self.trans(grid_world)



def collectFunc(batch: List[torch.Tensor]) -> List[Tensor]:
    return batch[0]  # (batch_size, 3, 127, 127)


class PathFinder:
    """
    An AI path finder in grid world, just take an action in [up, right, down, left, idle] every step

    Game rule:
    Agent is allowed to step on obstacle, and it allowed to stay at target
    If the agent step to obstacle or stay at obstacle, -5 reward is given
    If the agent reach or stay at the target, 1 reward is given
    If the agent stays at where it is, -0.1 reward is given
    If the agent go out of the world, 0 reward is given

    In fact, the agent cannot hit obstacle, but in some cases, the ego vehicle or the target point is very close to
    obstacles detected by lidar, and the ego vehicle or target point will appear covered or surrounded by obstacles,
    makeing the AI impossible to find a path to it, allowing passing through obstacles so the AI can find a path to 
    the target, while minimize the stepping on obstacles.

    Actions:
    The action contains UP, RIGHT, DOWN, LEFT, and IDLE

    """
    def __init__(self, configs):
        # --- Load dataset ---
        self.dataset = GridWorldDataset(configs["data_folders"])
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collectFunc)

        # --- Define policy model, target model and optimizers ---
        self.model: LocalPathPlanner = LocalPathPlanner().cuda()
        loadModel(self.model, RESUME)
        self.target_model = LocalPathPlanner().cuda()
        copyModel(self.model, self.target_model)
        self.optimizer_critic = torch.optim.Adam([{'params': self.model.critic.parameters()},
                                                  {'params': self.model.shared.parameters()}], lr=CRITIC_LR)
        self.optimizer_actor = torch.optim.Adam(self.model.actor.parameters(), lr=ACTOR_LR)

        self._initStepFilter()


    def _initStepFilter(self):
        """ This function initializes 5 conv layers, by applying conv layer to agent channel. 
        agent_channel = grid_world[:, 1, ...].unsqueeze(1), 
        the output will be the next state of the grid world environment.
        In this way, stepping can be embedded into neural networks.
        """
        # step_filter weight shape: (1, 1, 3, 3)
        # after applying corresponding weight to the grid_world[:, 1, ...], the result is the next step
        self.step_filters = [nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to("cuda") for _ in range(5)]

        up_weight = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device="cuda")
        up_weight[:, :, 2, 1] = 1
        self.step_filters[0].weight.data = up_weight

        right_weight = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device="cuda")
        right_weight[:, :, 1, 0] = 1
        self.step_filters[1].weight.data = right_weight

        down_weight = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device="cuda")
        down_weight[:, :, 0, 1] = 1
        self.step_filters[2].weight.data = down_weight

        left_weight = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device="cuda")
        left_weight[:, :, 1, 2] = 1
        self.step_filters[3].weight.data = left_weight

        idle_weight = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device="cuda")
        idle_weight[:, :, 1, 1] = 1
        self.step_filters[4].weight.data = idle_weight


    def step(self, grid_world: torch.Tensor, action: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step grid world to next state

        grid_world: (1, 3, 21, 21)
        grid_world[0, 0]: value maps
        grid_world[0, 1]: agent location
        grid_world[0, 2]: obstacles

        actions: [up, right, down, left, idle]

        reward(out of world): 0
        reward(reach target): 1
        reward(walk on obstacle): -1

        is_done: only true if out of world
        """
        agent_channel = grid_world[:, 1, ...].unsqueeze(1)  # (B, 1, 21, 21)
        next_agent_channel = self.step_filters[action](agent_channel)  # (B, 1, 21, 21)
        next_grid_world = grid_world.clone()
        next_grid_world[:, 1, ...] = next_agent_channel.squeeze(1)
        # uncomment this line if the passed by cells become obstacles
        # next_grid_world[:, 2, ...] += agent_channel.squeeze(1)

        # The agent arrived target location then the game is done
        # if agent is at target, then reached
        is_reached = torch.any((next_grid_world[:, 1, ...] + next_grid_world[:, 0, ...]).flatten(1) == 2).to(torch.float32)
        # if agent is at obstacle, then hit
        is_hit = torch.any((next_grid_world[:, 1, ...] + next_grid_world[:, 2, ...]).flatten(1) == 2).to(torch.float32)
        is_out = torch.all(next_grid_world[:, 1, ...].flatten(1) == 0).to(torch.float32)

        reward = is_reached - 5 * is_hit

        # grid_world[:, 1] is a mask of agent location
        # grid_world[:, 0] is value map, pixel value corresponds to the value of agent being at this position
        # they multiplied together only remains one value, the value of where the agent is at
        # This reward is positive if the agent move towards the target,
        # and it is negative if the agent is moving away from the target
        moving_reward = (torch.sum(next_grid_world[:, 1] * next_grid_world[:, 0]) -
                         torch.sum(grid_world[:, 1] * grid_world[:, 0]))
        reward += 0.01 * torch.sign(moving_reward)
        return next_grid_world, reward, is_out


    def visualize(self, mem_tuple, t, wait_time=1):
        grid_world, action, reward, next_grid_world, is_done = mem_tuple
        grid_world_np = grid_world[0].permute(1, 2, 0).detach().cpu().numpy()
        reward_f = float(reward.detach().cpu())
        next_grid_world_np = next_grid_world[0].permute(1, 2, 0).detach().cpu().numpy()
        temp = cv2.resize(np.hstack([grid_world_np, next_grid_world_np]), dsize=None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        cv2.putText(temp, f"r={reward_f:.3f} t={t:4d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("memory tuple", temp)
        return cv2.waitKey(wait_time)


    def train(self):
        # Memory
        memory = []
        mem_size = 20000
        target_model_update_freq = 4000
        random_prob = RAND_PROB
        greedy_prob = GREEDY_PROB
        gamma = GAMMA
        train_it = 0
        env_it = 0
        max_steps = MAX_STEPS

        # Log writers
        summary_writer = SummaryWriter(log_dir="Checkpoints/PathFinder")
        train_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_writer = LogWriter(train_start_time, "Checkpoints/PathFinder")

        # Statistics record
        avg_count = 100
        avg_reward = MovingAverage(avg_count)
        avg_episode_result = MovingAverage(avg_count)
        avg_actor_loss = MovingAverage(avg_count)
        avg_critic_loss = MovingAverage(avg_count)

        # train for 10 epochs, but actually this is not necessary, this can be while(True)
        for epoch in range(10):
            # An episode_data is one single saved lidar map
            for episode, episode_data in enumerate(self.dataloader):
                # region Place one episode
                grid_world = episode_data

                episode_reward = 0
                n_steps = 0
                is_done = False
                self.model.eval()
                with torch.no_grad():
                    # Play for one episode until the grid world game is end, or maximum step reached
                    while (n_steps < max_steps and not is_done):
                        action = self.model.getAction(grid_world, random_prob, greedy_prob)
                        next_grid_world, reward, is_done = self.step(grid_world, action)

                        episode_reward += reward.item()

                        avg_reward.add(reward.item())
                        memory.append((
                            grid_world, action, reward, next_grid_world, is_done
                        ))
                        if len(memory) > mem_size:
                            memory.pop(0)

                        grid_world = next_grid_world
                        n_steps += 1
                        env_it += 1

                        if env_it % 2 == 0 or is_done.item() == 1:
                            self.visualize(memory[-1], n_steps)

                        if env_it % 500 == 0:
                            mean_reward = avg_reward.get()
                            log_writer.log("ENVIRONMENT", f"env_it = {env_it}, reward = {mean_reward:.4f}")
                            summary_writer.add_scalar("reward", mean_reward, env_it)

                avg_episode_result.add(episode_reward)

                if episode % 100 == 0:
                    log_writer.log("ENVIRONMENT", f"avg episode result: {avg_episode_result.get():.4f}")
                    summary_writer.add_scalar("episode reward", avg_episode_result.get(), env_it)
                
                # endregion

                if len(memory) < 10000:
                    log_writer.log("ENVIRONMENT", f"Filling memory, memory size: {len(memory)}")
                    continue

                self.model.train()
                # Train for MAX_STEPS times, because one episode is likely to have MAX_STEPS steps
                # which provides MAX_STEPS new data
                for train_step in range(MAX_STEPS):
                    batch = random.sample(memory, BATCH_SIZE)
                    states, actions, rewards, next_states, is_dones = zip(*batch)
                    states = torch.cat(states, dim=0)
                    actions = torch.tensor(actions, dtype=torch.int64, device="cuda")
                    rewards = torch.stack(rewards, dim=0)
                    next_states = torch.cat(next_states, dim=0)
                    is_dones = torch.stack(is_dones, dim=0)

                    self.optimizer_critic.zero_grad()
                    self.optimizer_actor.zero_grad()

                    with torch.no_grad():
                        _, V_next_s = self.target_model(next_states)  # (B)
                        td_target = (rewards + gamma * V_next_s * (1 - is_dones)).detach()   # (B)

                    policies: torch.distributions.Categorical
                    policies, V_s = self.model(states)  # (B, 4) and (B)

                    adv = td_target - V_s  # (B)

                    entropy = policies.entropy()

                    actor_loss = (-policies.log_prob(actions) * adv.detach() - 0.01 * entropy).mean()

                    critic_loss = adv.pow(2).mean()

                    loss = critic_loss + actor_loss

                    avg_actor_loss.add(actor_loss.item())
                    avg_critic_loss.add(critic_loss.item())

                    if torch.any(torch.isnan(loss)):
                        log_writer.log("ERROR", f"loss: {loss.item():.4f} actor: {actor_loss.item():.4f} critic: {critic_loss.item():.4f}")

                    loss.backward()

                    # nn.utils.clip_grad_norm_(self.model.actor.parameters(), 0.5)

                    self.optimizer_critic.step()
                    self.optimizer_actor.step()

                    train_it += 1

                    if train_it % target_model_update_freq == 0:
                        copyModel(self.model, self.target_model)

                    if train_it % 500 == 0:
                        log_writer.log("TRAIN", f"actor: {avg_actor_loss.get():.4f} critic: {avg_critic_loss.get():.4f}")
                        summary_writer.add_scalar("loss", loss.item(), train_it)
                        summary_writer.add_scalar("actor loss", avg_actor_loss.get(), train_it)
                        summary_writer.add_scalar("critic loss", avg_critic_loss.get(), train_it)

                    if train_it % 10000 == 0:
                        saveModel(self.model, f"Checkpoints/PathFinder/{train_it}.pth")


if __name__ == '__main__':
    with open("config_with_mem.yaml", 'r') as in_file:
        configs = yaml.load(in_file, Loader=yaml.FullLoader)

    # dataset = MemoryDataset(configs["data_folders"])
    # for i in range(0, len(dataset), 100):
    #     lidar_map = dataset.getSource(i)
    #     test_grid_world = dataset[i]
    #     grid_world_np = cv2.resize(test_grid_world[0].permute(1, 2, 0).cpu().numpy(), (512, 512), interpolation=cv2.INTER_NEAREST)
    #     lidar_map_np = cv2.resize(lidar_map[0].permute(1, 2, 0).cpu().numpy(), (512, 512), interpolation=cv2.INTER_NEAREST)
    #     cv2.imshow("lidar map", lidar_map_np)
    #     cv2.imshow("grid world", grid_world_np)
    #     cv2.waitKey()

    os.makedirs(f"Checkpoints/PathFinder", exist_ok=True)
    path_finder = PathFinder(configs)
    path_finder.train()
