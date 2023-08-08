"""
This script is used to evaluate the performance of different agents.
"""
from Agents.A2CAgent import A2CAgent, OnlyInferA2CAgent
from Agents.AgentBasic import AgentBasic, OnlyInferAgentBasic
from Agents.HumanAgent import HumanAgent
from Agents.IdleAgent import IdleAgent
from Agents.GreedyAgent import GreedyAgent
from Agents.PathAgent import PathAgent

from VehicleEnv import VehicleEnv
from TrainUtils import *
import cv2
import yaml
import time
import pickle
from TrainUtils import LogWriter
from datetime import datetime
import os


agent_dict = {
    "Greedy": GreedyAgent,
    "Human": IdleAgent,
    "Random": AgentBasic,
    "A2C": A2CAgent,
    "Path": PathAgent,
}

def buildAgent(agent_name: str, agent_config_path: str):
    with open(agent_config_path, 'r') as in_file:
        agent_config = yaml.load(in_file, Loader=yaml.FullLoader)

    if agent_name == "A2C":
        agent = A2CAgent(agent_config)
        only_infer_agent = OnlyInferA2CAgent(agent)
        only_infer_agent.greedy_prob = 0.1
        only_infer_agent.epsilon = 0.01
        only_infer_agent.repeat_mean = 5
    elif agent_name == "Greedy":
        only_infer_agent = GreedyAgent(agent_config)
        only_infer_agent.epsilon = 0.05
        only_infer_agent.repeat_mean = 5
    elif agent_name == "Path":
        only_infer_agent = PathAgent(agent_config)
        only_infer_agent.epsilon = 0.00
        only_infer_agent.greedy_prob = 0.0
        only_infer_agent.repeat_mean = 0
    elif agent_name == "Random":
        agent = AgentBasic(agent_config)
        only_infer_agent = OnlyInferAgentBasic(agent)
    elif agent_name == "Human":
        only_infer_agent = IdleAgent(agent_config)


    return only_infer_agent



def getAction(human_agent, model, state):
    human_action, control_signal_received, key_pressed = human_agent.getAction()
    if control_signal_received:
        return human_action

    V_s, throttle_brake_mu, steer_mu, heatmaps = model(*state.getTensor())
    temp = cv2.resize(heatmaps[0, 0].cpu().numpy(), dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("heatmap", temp)
    cv2.waitKey(1)
    return VehicleAction(throttle_brake_mu, steer_mu)



@torch.no_grad()
def runTestRoute(env: VehicleEnv, only_infer_agent, n_trials: int) -> Tuple[float, float, float, float, float]:
    """Run single test case

    Args:
        env (VehicleEnv): well prepared environment with route loaded
        only_infer_agent: The agent that mush have getAction method take VehicleState as input
        n_trials (int): number of trials to run

    Returns:
        time_spent (float): average time spent in each trial
        collision_counts (int): average number of collisions in each trial
        completion_percentage (float): average completion percentage in each trial
        total_rewards (float): average total rewards in each trial
    """


    steps_spent = np.zeros(n_trials, dtype=np.float32)
    collision_counts = np.zeros(n_trials, dtype=np.int32)
    completion_percentage = np.zeros(n_trials, dtype=np.float32)
    total_rewards = np.zeros(n_trials, dtype=np.float32)
    norm_scores = np.zeros(n_trials, dtype=np.float32)

    for i in range(n_trials):
        state = VehicleState(env.lidar_map, env.gnss_xyz, env.getNextTargetPoint(), env.compass, env.smooth_speed)
        action = only_infer_agent.getAction(state)
        is_done = False
        # Episode loop
        while not is_done:

            next_state, reward, done, user_input = env.step(action)
            steps_spent[i] += 1

            if user_input == ord("q"):
                env.destroy()
                cv2.destroyAllWindows()
                return 0, 0, 0, 0, 0

            if done:
                break
            else:
                state = next_state
                action = only_infer_agent.getAction(state)


        collision_counts[i] = env.collision_count
        completion_percentage[i] = env.completion_percentage
        total_rewards[i] = env.total_reward
        norm_scores[i] = env.normalized_score
        env.reset()

    return steps_spent.mean(), collision_counts.mean(), completion_percentage.mean(), total_rewards.mean(), norm_scores.mean()


def runEvaluation(test_config):

    # load agent config, this config may also include some environment settings
    # but only agent and model related items will be used
    only_infer_agent = buildAgent(test_config["test_agent"], test_config["agent_config_path"])

    n_trials = test_config["n_trials"]

    # Initialize log writer
    log_dir = os.path.join(test_config["log_dir"], datetime.now().strftime("%Y%m%d-%H%M%S-Eval"))
    os.makedirs(log_dir, exist_ok=True)

    log_writer = LogWriter(logger_title="Evaluation", log_dir=log_dir)

    log_writer.log("info", "Test Agent: {}".format(test_config["test_agent"]))
    log_writer.log("info", "Number of trials: {}".format(n_trials))

    test_case_dir = test_config["test_case_dir"]

    test_case_file = os.path.join(test_case_dir, "config.yaml")
    # test_case is a yaml file path, it contains environment configs
    # This config is likely to be the same as the agent config above
    with open(test_case_file, "rb") as in_file:
        test_case_config = yaml.load(in_file, Loader=yaml.FullLoader)

    with open(test_config["agent_config_path"], "rb") as in_file:
        env_config = yaml.load(in_file, Loader=yaml.FullLoader)
    env_config["world"] = test_case_config["world"]
    env_config["point_sparsity"] = test_case_config["point_sparsity"]
    env_config["route_path"] = test_case_config["route_path"]
    env_config["n_other_actors"] = test_case_config["n_other_actors"]

    log_writer.log("info", "Test case: {}".format(test_case_dir))

    # Initialize environment
    env = VehicleEnv(env_config)
    env.reset()
    log_writer.log("info", f"Route length: {env.route_size}")
    steps_spent, collision_count, completion_percentage, total_reward, norm_score = runTestRoute(env, only_infer_agent, n_trials)
    env.destroy()

    log_writer.log("result", "Average steps per point: {}".format(steps_spent / env.route_size))
    log_writer.log("result", "Average collisions per point: {}".format(collision_count / env.route_size))
    log_writer.log("result", "Average completion percentage: {}".format(completion_percentage))
    log_writer.log("result", "Average total rewards: {}".format(total_reward))
    log_writer.log("result", "Average rewards per point: {}".format(norm_score))



if __name__ == '__main__':
    with open("config_eval.yaml", 'r') as in_file:
        test_config = yaml.load(in_file, Loader=yaml.FullLoader)

    runEvaluation(test_config)
