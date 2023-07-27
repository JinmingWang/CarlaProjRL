# CarlaProjRL
Carla Project RL End-to-End Auto Driving

## Quick Start

```
$ git clone https://github.com/JinmingWang/CarlaProjRL && cd CarlaProjRL
$ git submodule init
$ git submodule update

$ conda create -n carla-rl-pytorch python=3.8 torchvision pytorch==1.13 cudatoolkit==11.6 cudnn -c pytorch -c nvidia
$ conda activate carla-rl-pytorch
$ pip install carla==0.9.14 pyyaml pynvml tensorboard
$ cd leaderboard && pip install -r requirements.txt && cd ..
$ cd scenario_runner && pip install -r requirements.txt && cd ..

$ python Train.py
```

## Program Entrance
- Run Train.py to start training (while interacting with CARLA), change config_1.yaml for setting
- Run TrainWithMemory.py to train with existing memory saved on disk, change config_with_mem.yaml for setting
- Change Models/config_model.yaml for model setting

## Useful Files
| File                      | Description                                                                                                        |
|---------------------------|--------------------------------------------------------------------------------------------------------------------|
| EnvUtils.py               | Defines VehicleAction and VehicleState                                                                             |
| Train.py                  | Training while interacting with CARLA environment, in multiprocessing mode. Running memories will be saved to disk |
| TrainWithMemory.py        | Training with existing memory saved on disk, in single process mode                                                |
| VehicleEnv_custom.py      | Defines the environment for training, it is an interface between training agents and CARLA environment             |
| TrainUtils.py             | Defines training utilities, loading, saving, copying of models, memory lists                                       |
| config_xxx.yaml             | Running config for Train.py                                                                                        |
| Agents/A2CAgent.py        | A2C agent that update the model and output actions                                                                 |
| Agents/HumanAgent.py      | It displays a control board for human to control the ego vehicle                                                   |
| Models/ModelUtils.py      | Defines some basic pytorch modules                                                                                 |
| Models/LidarModelSmall.py        | Defines a small model                                                                          |
| Models/LidarModelLarge.py | Defines a larger model                                                                       |
| Models/LidarModelRes.py    | Defines a model with resnet backbone                                                                                   |

# TODO
- The model is not learning, A2CAgent may contain potential problems.