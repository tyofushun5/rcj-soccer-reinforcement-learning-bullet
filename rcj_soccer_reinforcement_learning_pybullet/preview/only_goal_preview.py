import os
import time

from sb3_contrib import RecurrentPPO
from rcj_soccer_reinforcement_learning_pybullet.environment.only_ball_environment import OnlyBallGoalEnvironment

save_dir = "../model"

def main():
    preview_env = OnlyBallGoalEnvironment(max_steps=10000,
                                  create_position=[4.0, 0.0, 0.0],
                                  magnitude=10.0,
                                  gui=True)

    model_path = os.path.join(save_dir, "dispersion_only_goal_model_v1")
    loaded_model = RecurrentPPO.load(model_path, env=preview_env)

    observation, info = preview_env.reset()
    while True:
        action, _states = loaded_model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = preview_env.step(action)
        if terminated or truncated:
            preview_env.reset()

    # preview_env.close()

if __name__ == "__main__":
    main()
