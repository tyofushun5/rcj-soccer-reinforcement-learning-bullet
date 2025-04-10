import os

from typing import Tuple
import numpy as np
from sb3_contrib import RecurrentPPO
import torch as th
import torch.nn as nn
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from rcj_soccer_reinforcement_learning_pybullet.environment.environment import Environment


script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
model_path = os.path.join(parent_dir, 'model', 'default_model', 'default_model_v1')


class OnnxableSB3Policy(nn.Module):
    def __init__(self, policy: RecurrentActorCriticPolicy):
        super().__init__()
        self.policy = policy

    def forward(
            self,
            observation: th.Tensor,
            pi: th.Tensor,
            vf: th.Tensor,
            episode_start: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:

        return self.policy(
            observation,
            RNNStates(
                pi=pi,
                vf=vf,
            ),
            episode_start,
            deterministic=True,
        )


def main():
    model = RecurrentPPO.load(path=model_path, device="cpu")

    observation_size = model.observation_space.shape
    dummy_observation = np.zeros((1, *observation_size), dtype=np.float32)
    dummy_episode_starts = np.ones((1,), dtype=bool)

    _, dummy_lstm_states = model.predict(
        observation=dummy_observation,
        state=None,
        episode_start=dummy_episode_starts,
        deterministic=True,
    )

    print(f"dummy_lstm_states: {dummy_lstm_states[0]}")

    th.onnx.export(
        model=OnnxableSB3Policy(model.policy),
        args=(
            th.randn((1, *observation_size), dtype=th.float32),
            th.from_numpy(np.array([dummy_lstm_states[0], dummy_lstm_states[1]])),
            th.from_numpy(np.array([dummy_lstm_states[0], dummy_lstm_states[1]])),
            th.ones((1,), dtype=th.float32),
        ),
        f="checkpoints/robocup.onnx",
    )

    env = Environment(
        max_steps=5000,
        create_position=[4.0, 0.0, 0.0],
        magnitude=21.0,
        gui=True
    )

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    obs = env.reset()
    for _ in range(1000):
        actions, lstm_states = model.predict(
            observation=obs[0],
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        obs, reward, done, _ = env.step(actions=np.array([actions]))

        episode_starts = done

        print(f"actions: {actions}")
        print(f"obs: {obs[0]}")
        print(f"reward: {reward[0]}")
        print(f"done: {done[0]}")

        if done:
            env.reset()


if __name__ == "__main__":
    main()
