import os

from sb3_contrib import RecurrentPPO
from rcj_soccer_reinforcement_learning_pybullet.environment.only_ball_environment import OnlyBallGoalEnvironment

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
model_path = os.path.join(parent_dir, "model",'only_goal_model', "dispersion_only_goal_model_v1")

def main():
    preview_env = OnlyBallGoalEnvironment(max_steps=10000,
                                  create_position=[4.0, 0.0, 0.0],
                                  magnitude=0.1,
                                  gui=True)

    loaded_model = RecurrentPPO.load(model_path, env=preview_env)

    observation, info = preview_env.reset()
    while True:
        action, _states = loaded_model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = preview_env.step(action)
        print(action)
        if terminated or truncated:
            preview_env.reset()

    # preview_env.close()

if __name__ == "__main__":
    main()
