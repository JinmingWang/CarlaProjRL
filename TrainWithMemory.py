# Train data with memory loaded from disk instead of interacting with environment

import os
from Agents.A2CAgent import A2CAgent
import yaml
import multiprocessing
from TrainUtils import *
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2

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

    def __getitem__(self, idx):
        memory_tuple = MemoryTuple.fromTuple(torch.load(self.file_paths[idx]))
        return memory_tuple



def collectFunc(batch: List[MemoryTuple]) -> Tuple:
    return MemoryTuple.makeBatch(batch)



def train(configs, n_epochs: int = 40):
    program_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    configs["save_dir"] = os.path.join(configs["save_dir"], program_start_time) + "/"
    configs["log_dir"] = os.path.join(configs["log_dir"], program_start_time) + "/"
    os.mkdir(configs["save_dir"])
    os.mkdir(configs["log_dir"])

    summary_writer = SummaryWriter(configs["log_dir"])

    dataset = MemoryDataset(configs["data_folders"])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collectFunc)

    agent = A2CAgent(configs)

    loss_records = MovingAverage(configs["moving_average_window"])
    actor_loss_records = MovingAverage(configs["moving_average_window"])
    critic_loss_records = MovingAverage(configs["moving_average_window"])

    it = 0

    for epoch_i in range(n_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_i}")
        for batch_i, batch_data in enumerate(pbar):

            loss, critic_loss, actor_loss = agent.trainStep(batch_data)
            loss_records.add(loss)
            actor_loss_records.add(actor_loss)
            critic_loss_records.add(critic_loss)

            pbar.set_postfix_str(f"loss: {loss_records.get():.4f}, actor_loss: {actor_loss_records.get():.4f}, critic_loss: {critic_loss_records.get():.4f}")

            if it % configs["show_freq"] == 0:
                policy_map = agent.policy_map
                policy_map /= policy_map.max()
                cv2.imshow("policy map", policy_map)
                cv2.waitKey(1)

            if it % configs["log_freq"] == 0:
                summary_writer.add_scalar("loss", loss_records.get(), it)
                summary_writer.add_scalar("actor_loss", actor_loss_records.get(), it)
                summary_writer.add_scalar("critic_loss", critic_loss_records.get(), it)

            it += 1

        saveModel(agent.model, os.path.join(configs["save_dir"], f"model_epoch{epoch_i}.pth"))


def prepare() -> Dict:
    with open("config_with_mem.yaml", 'r') as in_file:
        configs = yaml.load(in_file, Loader=yaml.FullLoader)
    return configs


if __name__ == "__main__":
    configs = prepare()
    train(configs)