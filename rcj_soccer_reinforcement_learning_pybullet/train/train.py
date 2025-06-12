import os

from stable_baselines3 import PPO
from rcj_soccer_reinforcement_learning_pybullet.environment.environment import Environment

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
save_dir = os.path.join(parent_dir, 'model','default_model')
os.makedirs(save_dir, exist_ok=True)

def main():
    # 環境の作成
    env = Environment(create_position=[4.0, 0.0, 0.0],
                          max_steps=10000,
                          magnitude=21.0,
                          gui=True)

    policy_kwargs = {
        "net_arch": dict(pi=[256, 256, 256], vf=[256, 256, 256]),
    }

    model = PPO(
        'MlpPolicy',
        env,
        device='cuda',
        verbose=1,
        n_epochs=10,
        n_steps=2048,
        batch_size=2048,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        max_grad_norm=0.5
    )

    model.learn(total_timesteps=5000000)
    
    model.save(os.path.join(save_dir, 'default_model_v10'))

    env.close()

if __name__ == '__main__':
    main()
