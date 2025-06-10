import random
import math

import numpy as np
import torch
import torch.nn as nn
import torchrl
from torchrl.envs import EnvBase
from torchrl.data import Unbounded, Composite, Bounded, Binary
from tensordict import TensorDict
import pybullet as p

class Environment(EnvBase):
    def __init__(self):
        super().__init__()
        #Agentの観測空間
        self.observation_space = Composite(
            normalized_ball_angle = Bounded(low=-1.0, high=1.0, shape=(1,), dtype=torch.float32),
            normalized_enemy_goal_angle = Bounded(low=-1.0, high=1.0, shape=(1,), dtype=torch.float32),
            is_online = Binary(shape=(1,), dtype=torch.float32),
        )
        #Agentの行動空間
        self.action_space = Composite(
            normalized_x_axis_vector = Bounded(low=-1.0, high=1.0, shape=(1,), dtype=torch.float32),
            normalized_y_axis_vector = Bounded(low=-1.0, high=1.0, shape=(1,), dtype=torch.float32),
            normalized_theta = Bounded(low=-1.0, high=1.0, shape=(1,), dtype=torch.float32),
        )

    def _reset(self, tensordict):
        pass

    def _step(self, tensordict):
        out = TensorDict(
            {
                "reward": torch.tensor(0.0, dtype=torch.float32),
                "done": torch.tensor(False, dtype=torch.bool),
            },
            tensordict.shape
        )
        return out

    def _set_seed(self):
        torch.manual_seed(0)

