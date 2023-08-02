import multiprocessing
import os
import time
from Agents.AgentBasic import AgentBasic, OnlyInferAgentBasic
from Agents.A2CAgent import A2CAgent, OnlyInferA2CAgent
# from VehicleEnv_official import VehicleEnv
from VehicleEnv import VehicleEnv
import yaml
import multiprocessing
from TrainUtils import *
from torch.utils.tensorboard import SummaryWriter
import cv2
import pynvml


def runEnvironment(configs, agent, avg_reward, shared_model, shared_model_update, cache, end_signal, save_memory_signal):
    """
    This function will be executed in a separate process
    1. Initialize environment
    2. Initialize shadow agent
        A shadow agent is only used to get action from state, it does not train.
        This agent is only visible to the environment process.
        Train process will periodically copy the latest trained model to self.shared_model.obj.
        We will then update shadow agent's model using the shared model.
    3. Simulation for infinite episodes
    4. Simulation an episode till done, and just add all memory tuples to cache queue
    """
    logger = LogWriter("Environment", configs["log_dir"])

    logger.log("info", "Initializing Environment...")

    env = VehicleEnv(configs)
    env.reset()

    reward_records = MovingAverage(configs["moving_average_window"])

    shadow_agent = OnlyInferA2CAgent(agent)

    logger.log("info", "Environment Initialized, Simulation Start...")

    for k, v in configs.items():
        logger.log("config", f"{k}={v}")

    n_iter = 0

    # Simulation loop
    while True:
        state = VehicleState(env.lidar_map, env.gnss_xyz, env.getNextTargetPoint(), env.compass, env.smooth_speed)
        action = shadow_agent.getAction(state)
        is_done = False
        # Episode loop
        while not is_done:
            next_state, reward, done, user_input = env.step(action)
            reward_records.add(reward)
            avg_reward.value = reward_records.get()

            # Keep adding memory tuples to cache queue
            # but once every 4 iterations
            if n_iter % 4 == 0:
                record = MemoryTuple()
                record.state = state
                record.action = action
                record.next_state = next_state
                record.reward = reward
                record.done = done
                cache.put(record)

            n_iter += 1
            if n_iter % (configs["memory_save_freq"] * 4) == 0:
                save_memory_signal.value = n_iter // configs["memory_save_freq"]

            if user_input == ord("r"):
                break
            elif user_input == ord("q"):
                logger.log("info", "Environment process received quit signal, exiting...")
                env.destroy()
                end_signal.value = 1
                cv2.destroyAllWindows()
                return

            if done:
                collision_count = env.collision_count
                total_reward = env.total_reward
                percentage_completion = env.completion_percentage
                norm_score = env.normalized_score
                route_length = env.route_size
                logger.log("info", f"Episode End, Collision Count: {collision_count}, Total Reward: {total_reward}, "
                                   f"Percentage Completion: {percentage_completion}, Normalized Score: {norm_score}, "
                                   f"Route Length: {route_length}")
                break
            else:
                state = next_state
                action = shadow_agent.getAction(state)

            # check for the need of update model
            # if needed, copy model to shared model as CPU and set flag to 0
            if shared_model_update.value != 0:
                shared_model_update.value = 0
                shadow_agent.model.cpu()
                copyModel(shared_model.obj, shadow_agent.model)
                shadow_agent.model.to(shadow_agent.device)

        env.reset()


def trainLoop():
    """
    This function will be executed in the main process

    1. Move data from cache queue to memory list
    2. Sample batch data from memory list
    3. Train agent using batch data
    4. Copy agent model to shared model as CPU model and set flag to 1 to notify environment process
    5. Repeat and save model and log
    :return:
    """
    # sample batch data
    # send to agent.train_step to train
    memory = MemoryList(configs["memory_size"])
    summary_writer = SummaryWriter(configs["log_dir"])
    loss_records = MovingAverage(configs["moving_average_window"])
    logger = LogWriter("Train", configs["log_dir"])
    it = 0

    dynamic_batch_size = 1

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    logger.log("info", "Waiting for environment to fill memory...")
    while len(memory) < configs["batch_size"]:
        while not cache.empty():
            memory.append(cache.get())
    logger.log("info", "Training Start...")

    while True:
        if end_signal.value == 1:
            logger.log("info", "Training End...")
            cv2.destroyAllWindows()
            return
        # move environment cache to train data memory
        while not cache.empty():
            memory.append(cache.get())

        if save_memory_signal.value != 0:
            logger.log("info", "Saving memory...")
            save_dir = f"/media/jimmy/MyData/Data/carla/{program_start_time}/{save_memory_signal.value}"
            os.makedirs(save_dir, exist_ok=True)
            memory.save(save_dir)
            save_memory_signal.value = 0

        # wait environment to fill memory
        if len(memory) < configs["batch_size"]:
            continue

        # (*state_tensors, action_tensor, reward_tensor, *next_state_tensors, done_tensor)
        # batch size will start from 1, and will grow to batch_size in several iterations
        # batch_tensors = memory.sampleBatch(min((it//10)+1, configs["batch_size"]))
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if meminfo.free / 1048576 >= 512 and dynamic_batch_size < configs["batch_size"] and it >= 100:
            dynamic_batch_size += 1
        batch_tensors = memory.sampleBatch(dynamic_batch_size)

        loss, message = agent.trainStep(batch_tensors)
        loss_records.add(loss)

        if message != "":
            logger.log("info", message)

        # copy agent model to shared model as CPU model
        # and set flag to 1 to notify environment process
        if it % configs["env_model_update_freq"] == 0:
            agent.model.cpu()
            copyModel(agent.model, shared_model.obj)
            shared_model_update.value = it
            agent.model.to(agent.device)


        if it % configs["log_freq"] == 0:
            summary_writer.add_scalar("loss", loss_records.get(), it)
            summary_writer.add_scalar("avg_reward", avg_reward.value, it)
            logger.log("info", f"Iteration {it}, loss: {loss_records.get()}, avg_reward: {avg_reward.value}, "
                               f"memory_size: {len(memory)}, batch_size: {dynamic_batch_size}")

        if it % configs["save_freq"] == 0:
            saveModel(agent.model, os.path.join(configs["save_dir"], f"model_it{it}.pth"))
            logger.log("info", f"Model saved to {configs['save_dir']}/model_it{it}.pth at iteration {it}")
        it += 1


def prepare() -> Dict:
    try:
        multiprocessing.set_start_method('spawn')  # have to set this to spawn, because subprocesses will use cuda
    except RuntimeError:
        print("RuntimeError: set_start_method('spawn') failed, maybe you have already set it.")

    with open("config_dense_lidar.yaml", 'r') as in_file:
        configs = yaml.load(in_file, Loader=yaml.FullLoader)
    return configs


if __name__ == '__main__':
    configs = prepare()

    program_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    LogWriter.start_time = program_start_time
    configs["save_dir"] = os.path.join(configs["save_dir"], program_start_time) + "/"
    configs["log_dir"] = os.path.join(configs["log_dir"], program_start_time) + "/"
    os.mkdir(configs["save_dir"])
    os.mkdir(configs["log_dir"])

    agent = A2CAgent(configs)

    # Process Shared Variables
    manager = multiprocessing.Manager()
    shared_model_update = manager.Value('i', 0)
    end_signal = manager.Value('i', 0)
    save_memory_signal = manager.Value('i', 0)
    shared_model = manager.Namespace()
    shared_model.obj = agent.createModel().cpu().eval()
    cache = multiprocessing.Queue(maxsize=configs["cache_size"])
    avg_reward = multiprocessing.Value('f', 0.0)

    subp = multiprocessing.Process(target=runEnvironment,
                                   args=(configs, agent, avg_reward, shared_model, shared_model_update, cache, 
                                         end_signal, save_memory_signal))
    subp.start()

    trainLoop()

    subp.terminate()
    subp.join()

    exit(0)
