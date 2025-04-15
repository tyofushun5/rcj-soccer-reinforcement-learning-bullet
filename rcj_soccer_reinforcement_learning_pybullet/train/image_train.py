import os

from sb3_contrib import RecurrentPPO
from rcj_soccer_reinforcement_learning_pybullet.environment.image_environment import ImageEnvironment

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

save_dir = os.path.join(parent_dir, "model")
os.makedirs(save_dir, exist_ok=True)

def main():

    env = ImageEnvironment(create_position=[4.0, 0.0, 0.0],
                          max_steps=10000,
                          magnitude=10.0,
                          gui=True)

    policy_kwargs = {
        "net_arch": dict(pi=[256, 256, 256], vf=[256, 256, 256]),
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True
    }

    model = RecurrentPPO(
        'MlpLstmPolicy',
        env,
        device='cuda',
        verbose=1,
        n_epochs=10,
        n_steps=128,
        batch_size=128,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        max_grad_norm=0.5
    )

    model.learn(total_timesteps=10000000)

    model.save(os.path.join(save_dir, "image_model_v1"))

    env.close()

if __name__ == "__main__":
    main()
