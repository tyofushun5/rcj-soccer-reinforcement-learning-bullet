import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from rcj_soccer_reinforcement_learning_pybullet.environment.goal_environment import GoalEnvironment
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback


script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
save_dir = os.path.join(parent_dir, "model",'goal_model')


checkpoint_callback = CheckpointCallback(save_freq=100000,
                                         save_path=save_dir,
                                         name_prefix='dispersion_goal_model_v1',
                                         save_replay_buffer=True,
                                         save_vecnormalize=True)

def make_env():
    def _init():
        env = GoalEnvironment(max_steps=10000,
                               create_position=[4.0, 0.0, 0.0],
                               magnitude=9.0,
                               gui=False)
        # check_env(env)
        return env
    return _init

def main():
    os.makedirs(save_dir, exist_ok=True)

    num_envs = 12
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = RecurrentPPO("MlpLstmPolicy",
                        env,
                        device="cuda",
                        verbose=1,
                        n_epochs=10,
                        n_steps=128,
                        batch_size=128*num_envs,
                        gamma=0.99)

    model.learn(total_timesteps=10000000)
    model.save(os.path.join(save_dir, "dispersion_goal_model_v1"))

    env.close()

if __name__ == "__main__":
    main()





