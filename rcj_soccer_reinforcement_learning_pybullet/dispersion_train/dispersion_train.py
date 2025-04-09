import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from rcj_soccer_reinforcement_learning_pybullet.environment.environment import Environment
from stable_baselines3.common.callbacks import CheckpointCallback

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
save_dir = os.path.join(parent_dir, 'model','default_model')
os.makedirs(save_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(save_freq=1000000,
                                         save_path=save_dir,
                                         name_prefix='default_model_v1',
                                         save_replay_buffer=True,
                                         save_vecnormalize=True)

def make_env():
    def _init():
        env = Environment(max_steps=5000,
                          create_position=[4.0, 0.0, 0.0],
                          magnitude=21.0,
                          gui=False)

        return env
    return _init

def main():

    num_envs = 12
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    policy_kwargs = {
        "net_arch": dict(pi=[256, 256, 256], vf=[256, 256, 256]),
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True
    }


    model = RecurrentPPO('MlpLstmPolicy',
                         env,
                         device='cuda',
                         verbose=1,
                         n_epochs=10,
                         n_steps=128,
                         batch_size=128*num_envs,
                         gamma=0.99,
                         policy_kwargs=policy_kwargs
                         )

    model.learn(total_timesteps=10000000,
                callback=checkpoint_callback,
                progress_bar=True
                )

    model.save(os.path.join(save_dir, 'default_model_v1'))

    env.close()

if __name__ == '__main__':
    main()





