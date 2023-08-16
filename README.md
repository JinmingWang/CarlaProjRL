# CarlaProjRL
Carla Project RL End-to-End Auto Driving with Reinforcement Learning + Imitation Learning

> https://github.com/JinmingWang/CarlaProjRL

### Project by Jinming Wang

# Key Features
- Carla simulator
- Reinforcement Learning, A2C
- Imitation Learning combined with RL
- Very simple path planning with grid world
- Point following
- Obstacle avoidance
- Random Repeat Random Actions
- Lidar sensor
- Point cloud to image, then to grid world

# Key Ideas
1. RL combined with IL to make the algorithm converge faster while does not need too much human demonstration
2. Process point cloud to lidar map image, then to grid world, with the ego vehicle at the center, a model is 
   trained using A2C to find the next action in the grid world {left, right, forward, backward, idle}, running this 
   model many times can generate a path from the agent to the target point.
3. Another model is trained using A2C to find the speed and steer action from the generated path drawn on the grid 
   world.
4. One single random action is applied N times for the next N steps to allow better exploration, N is a random number 
   with specified mean 
   and standard deviation.

# Quick Start

```
$ git clone https://github.com/JinmingWang/CarlaProjRL && cd CarlaProjRL
$ git submodule init
$ git submodule update

$ conda create -n carla-rl-pytorch python=3.8 torchvision pytorch==1.13 cudatoolkit==11.6 cudnn -c pytorch -c nvidia
$ conda activate carla-rl-pytorch
$ pip install carla==0.9.14 pyyaml pynvml tensorboard
$ cd leaderboard && pip install -r requirements.txt && cd ..
$ cd scenario_runner && pip install -r requirements.txt && cd ..

# Start Training (Carla server must be running):
$ python Train.py

# Train with existing saved memory on disk:
$ python TrainWithMemory.py

# Evaluation with test cases (Carla server must be running):
$ python Evaluate.py

# Play with environment (Carla server must be running):
$ python EnvRoute.py
# or
$ python EnvPracticeField.py
```

# Files

## Environment Files
| File                | Description                                                                   |
|---------------------|-------------------------------------------------------------------------------|
| EnvUtils.py         | Defines VehicleAction and VehicleState                                        |
| EnvRoute.py         | Defines the environment, the waypoints form a complete route                  |
| EnvPracticeField.py | Defines the environment, the next waypoint is a random point in defined areas |

## Agent Files
| File                 | Description                                                                |
|----------------------|----------------------------------------------------------------------------|
| Agents/A2CAgent.py   | A2C agent that update the model and output actions                         |
| Agents/HumanAgent.py | It displays a control board for human to control the ego vehicle           |
| Agents/AgentBasic.py | A basic agent that output random actions                                   |
| Agents/IdleAgent.py  | An agent that does nothing, usually used with human agent                  |
| Agents/GreedyAgent.py| Always go directly to the next waypoint (ignores obstacles)                |
| Agents/PathAgent.py  | Represent lidar map as grid world, run A* to find path and follow the path |

## Model Files
| File                    | Description                                                                                             |
|-------------------------|---------------------------------------------------------------------------------------------------------|
| Models/ModelUtils.py    | Defines some basic pytorch modules                                                                      |
| Models/LidarModel.py    | Directly output speed and steer action from lidar map and spatial vector                                |
| Models/LidarModelRes.py | Resnet version of the above model                                                                       |
| Models/PathModel.py     | This is a traditional algorithm, just run A* to find path, then compute speed and steer from thr path   |
| Models/RLPathModel.py   | Defines a model that finds the next step in grid world, and a model that outputs speed and steer action |

## Training and Evaluation Files
| File                     | Description                                                                                                         |
|--------------------------|---------------------------------------------------------------------------------------------------------------------|
| TrainUtils.py            | Defines training utilities, loading, saving, copying of models, memory lists                                        |
| PathFinder/PathFinder.py | The trainer for RL path finder using A2C, the well trained network can find the next step in grid world             |
| Train.py                 | Multiprocessing training code                                                                                       |
| TrainWithMemory.py       | Single process training code, with saved data                                                                       |
| Evaluate.py              | Evaluation code, with test case                                                                                     |
| WeaknessGather.py        | Apply trained model greedily, human can see and adjust agent's behavior, human control will be saved as new dataset |

## Config Files
| File                              | Description                                                   |
|-----------------------------------|---------------------------------------------------------------|
| config_dense_lidar.yaml           | Config file for EnvRoute, agent, traininga and testing        |
| config_dense_lidar_no_others.yaml | Same as above, but no other vehicles and pedestrians          |
| config_eval.yaml                  | Config file for evaluation                                    |
| config_practice_x.yaml            | Config file for EnvPracticeField, agent, training and testing |
| config_weakness_gather.yaml       | Config file for WeaknessGather.py                             |
| config_with_mem.yaml              | Config file for TrainWithMemory.py                            |


## Other Files
| File                      | Description                                                     |
|---------------------------|-----------------------------------------------------------------|
| TestCases/*               | Test cases for evaluation                                       |
| Logs/*                    | Logs for training and evaluation                                |
| Dairy/*                   | Logs and dairy writen by myself to record the project progress  |
| Checkpoints/date-time     | Saved models                                                    |
| Checkpoints/PathFinder/*  | Saved models and also training logs for PathFinder              |


# TODO
- The model is not learning, A2CAgent may contain potential problems.