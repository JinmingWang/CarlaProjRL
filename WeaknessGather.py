"""
The agent sometimes may face following problems:
1. Stuck in place
2. Apply suboptimal actions
3. Collision

It may be time-consuming to gather enough data to train the agent to solve these specific problems.
It takes lots of experience to learn to solve these problems.

This script works in the following way:
1. Run the environment with a currently well-trained agent
2. The agent meets a problem
3. The human expert takes control of the agent and solves the problem, human behaviors are recorded
4. Human behaviors are stored as dataset
"""
from Agents.A2CAgent import A2CAgent, OnlyInferA2CAgent
from Agents.AgentBasic import AgentBasic, OnlyInferAgentBasic
from Agents.HumanAgent import HumanAgent
from Agents.IdleAgent import IdleAgent
from Agents.GreedyAgent import GreedyAgent

from VehicleEnv import VehicleEnv
from TrainUtils import *
import cv2
import yaml
import time
import pickle
from TrainUtils import LogWriter
from datetime import datetime
import os


def runEnvironment():
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

    program_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    env = VehicleEnv(configs)
    env.reset()

    shadow_agent = OnlyInferA2CAgent(agent)

    mem_list = MemoryList(configs["memory_size"])

    n_iter = 0

    # Simulation loop
    while True:
        state = VehicleState(env.lidar_map, env.gnss_xyz, env.getNextTargetPoint(), env.compass, env.smooth_speed)
        action = shadow_agent.getAction(state)
        is_done = False
        # Episode loop
        while not is_done:
            next_state, reward, done, user_input = env.step(action)

            # Keep adding memory tuples to cache queue
            # but once every 4 iterations
            if action.is_human_action:
                record = MemoryTuple()
                record.state = state
                record.action = action
                record.next_state = next_state
                record.reward = reward
                record.done = done
                mem_list.append(record)


            n_iter += 1

            if n_iter % 1000 == 0:
                print(f"n_iter: {n_iter}, memory size: {len(mem_list)}")

            if mem_list.isFull() or user_input == ord("s"):
                os.makedirs(f"/media/jimmy/MyData/Data/carla/{program_start_time}/{n_iter:7d}", exist_ok=True)
                mem_list.save(f"/media/jimmy/MyData/Data/carla/{program_start_time}/{n_iter:7d}")
                mem_list = MemoryList(configs["memory_size"])
            if user_input == ord("r"):
                break
            elif user_input == ord("q"):
                env.destroy()
                cv2.destroyAllWindows()
                return

            if done:
                break
            else:
                state = next_state
                action = shadow_agent.getAction(state)

        env.reset()



if __name__ == '__main__':
    with open("config_dense_lidar.yaml", 'r') as in_file:
        configs = yaml.load(in_file, Loader=yaml.FullLoader)

    configs["epsilon"] = 0
    configs["n_repeat_rand_actions"] = 0
    agent = A2CAgent(configs)

    runEnvironment()
