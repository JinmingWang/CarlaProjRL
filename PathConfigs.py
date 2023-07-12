import os
import sys

"""
In case of some users have difficulty in setting up the environment, paths, packages, etc.
This file ensures the paths are included and packages can be imported.
"""

""" --- Define Paths --- """
PROJ_PATH = "/home/jimmy/Desktop/CARLA_PROJ"

CARLA_ROOT = "/home/jimmy/CARLA/CARLA_Leaderboard_20"
LEADERBOARD_ROOT = "/home/jimmy/CARLA/leaderboard"
SCENARIO_RUNNER_ROOT = "/home/jimmy/CARLA/scenario_runner"


""" --- Add Paths ---"""
sys.path.append(f"{CARLA_ROOT}/PythonAPI/carla")

sys.path.append(f"{CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg")
import carla

sys.path.append(f"{LEADERBOARD_ROOT}")
sys.path.append(f"{SCENARIO_RUNNER_ROOT}")

from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from leaderboard.autoagents.autonomous_agent import Track


if __name__ == '__main__':
    pass