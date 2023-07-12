# CarlaProjRL
Carla Project RL End-to-End Auto Driving

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
| config_1.yaml             | Running config for Train.py                                                                                        |
| config_with_mem.yaml      | Running config for TrainWithMemory.py                                                                              |
| Agents/A2CAgent.py        | A2C agent that update the model and output actions                                                                 |
| Agents/HumanAgent.py      | It displays a control board for human to control the ego vehicle                                                   |
| Models/ModelUtils.py      | Defines some basic pytorch modules                                                                                 |
| Models/SegModel.py        | Defines the RGB camera segmentation model                                                                          |
| Models/PointCloudModel.py | Defines the Lidar or Radar point cloud model                                                                       |
| Models/CompositeA2C.py    | Defines the multimodal A2C model                                                                                   |
| Models/config_model.yaml  | Model setting                                                                                                      |

# TODO
- The model is not learning, A2CAgent may contain potential problems.