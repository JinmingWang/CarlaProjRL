# Train data with memory loaded from disk instead of interacting with environment

import os

from torch import Tensor

from Agents.A2CAgent import A2CAgent
import yaml
from TrainUtils import *
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MemoryDataset(Dataset):
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

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx) -> Tuple[VehicleState, VehicleAction, float, VehicleState, bool]:
        # state, action, reward, next_state, done
        memory_tuple = torch.load(self.file_paths[idx])
        return memory_tuple


def collectFunc(batch: List[Tuple[VehicleState, VehicleAction, float, VehicleState, bool]]) -> List[Tensor]:
    """
    Pack dataset list data into a batch
    :param batch: A list of data loaded
    :return: Batch data that can be directly used in training
    """
    lidar_maps = []
    spatial_features = []
    actions = []
    rewards = []
    next_lidar_maps = []
    next_spatial_features = []
    dones = []
    human_action_mask = []
    for i, mem_tuple in enumerate(batch):
        lidar_map, spatial_feature = mem_tuple[0].getTensor()
        lidar_maps.append(lidar_map)
        spatial_features.append(spatial_feature)

        actions.append(mem_tuple[1].getTensor())

        rewards.append(mem_tuple[2])

        next_lidar_map, next_spatial_feature = mem_tuple[3].getTensor()
        next_lidar_maps.append(next_lidar_map)
        next_spatial_features.append(next_spatial_feature)

        dones.append(mem_tuple[4])

        human_action_mask.append(mem_tuple[1].is_human_action)

    return torch.cat(lidar_maps), torch.cat(spatial_features), \
        torch.cat(actions), \
        torch.tensor(rewards, dtype=torch.float32, device=MemoryDataset.device), \
        torch.cat(next_lidar_maps), torch.cat(next_spatial_features), \
        torch.tensor(dones, dtype=torch.bool, device=MemoryDataset.device), \
        torch.tensor(human_action_mask, dtype=torch.bool, device=MemoryDataset.device)


def train(configs, n_epochs: int = 40):
    program_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    configs["save_dir"] = os.path.join(configs["save_dir"], program_start_time) + "/"
    configs["log_dir"] = os.path.join(configs["log_dir"], program_start_time) + "/"
    os.mkdir(configs["save_dir"])
    os.mkdir(configs["log_dir"])

    summary_writer = SummaryWriter(configs["log_dir"])

    dataset = MemoryDataset(configs["data_folders"])
    dataloader = DataLoader(dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=0, collate_fn=collectFunc)

    agent = A2CAgent(configs)

    loss_records = MovingAverage(configs["moving_average_window"])
    policy_loss_records = MovingAverage(configs["moving_average_window"])
    value_loss_records = MovingAverage(configs["moving_average_window"])
    human_loss_records = MovingAverage(configs["moving_average_window"])

    it = 0

    for epoch_i in range(n_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_i}")
        for batch_i, batch_data in enumerate(pbar):

            loss, policy_loss, value_loss, human_loss, msg = agent.trainStep(batch_data)
            loss_records.add(loss)
            policy_loss_records.add(policy_loss)
            value_loss_records.add(value_loss)
            human_loss_records.add(human_loss)

            pbar.set_postfix_str(f"loss={loss:.4f}, " + msg)

            it += 1

            if it % configs["log_freq"] == 0:
                summary_writer.add_scalar("loss", loss_records.get(), it)
                summary_writer.add_scalar("policy_loss", policy_loss_records.get(), it)
                summary_writer.add_scalar("value_loss", value_loss_records.get(), it)
                summary_writer.add_scalar("human_loss", human_loss_records.get(), it)

            if it % configs["model_save_freq"] == 0:
                saveModel(agent.model, os.path.join(configs["save_dir"], f"model_{it}.pth"))


def prepare() -> Dict:
    with open("config_with_mem.yaml", 'r') as in_file:
        configs = yaml.load(in_file, Loader=yaml.FullLoader)
    return configs


if __name__ == "__main__":
    configs = prepare()
    train(configs)