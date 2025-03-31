import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from rcj_soccer_reinforcement_learning_pybullet.environment.environment import Environment
from stable_baselines3.common.callbacks import CheckpointCallback


script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
save_dir = os.path.join(parent_dir, 'model','default_model')
os.makedirs(save_dir, exist_ok=True)

def make_env():
    def _init():
        env = Environment(max_steps=10000,
                              create_position=[4.0, 0.0, 0.0],
                              magnitude=21.0,
                              gui=False)

        return env
    return _init

def main():

    num_envs = 12
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = RecurrentPPO('MlpLstmPolicy',
                         env,
                         device='cuda',
                         verbose=1,
                         n_epochs=10,
                         n_steps=256,
                         batch_size=256*num_envs,
                         gamma=0.99)

    model.learn(total_timesteps=50000000)
    model.save(os.path.join(save_dir, 'default_model_v1'))

    env.close()

if __name__ == '__main__':
    main()





