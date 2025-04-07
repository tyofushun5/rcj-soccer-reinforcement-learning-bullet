import os
import torch as th
import torch.nn as nn
from typing import Tuple

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.policies import BasePolicy
from rcj_soccer_reinforcement_learning_pybullet.environment.environment import Environment


script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
model_path = os.path.join(parent_dir, 'model', 'default_model', 'default_model_v1')


class OnnxableSB3Policy(nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return self.policy(observation, deterministic=True)

preview_env = Environment(
    max_steps=5000,
    create_position=[4.0, 0.0, 0.0],
    magnitude=21.0,
    gui=True
)

model = RecurrentPPO.load(model_path, env=preview_env)

onnx_policy = OnnxableSB3Policy(model.policy)

observation_size = model.observation_space.shape
dummy_input = th.randn(1, *observation_size)

th.onnx.export(
    onnx_policy,
    dummy_input,
    "my_ppo_model.onnx",
    opset_version=17,
    input_names=["input"],
)

import onnx
import onnxruntime as ort
import numpy as np

onnx_path = "my_ppo_model.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

observation = np.zeros((1, *observation_size)).astype(np.float32)
ort_sess = ort.InferenceSession(onnx_path)
actions, values, log_prob = ort_sess.run(None, {"input": observation})

print(actions, values, log_prob)

# Check that the predictions are the same
with th.no_grad():
    print(model.policy(th.as_tensor(observation), deterministic=True))