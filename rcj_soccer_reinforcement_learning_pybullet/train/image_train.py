import os

from sb3_contrib import RecurrentPPO
from rcj_soccer_reinforcement_learning_pybullet.environment.image_environment import ImageEnvironment

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

save_dir = os.path.join(parent_dir, "model")
os.makedirs(save_dir, exist_ok=True)

def main():
    # 環境の作成
    env = ImageEnvironment(create_position=[4.0, 0.0, 0.0],
                          max_steps=10000,
                          magnitude=10.0,
                          gui=False)

    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        device="cuda",
        verbose=1,
        n_epochs=10,
        n_steps=128,
        batch_size=128,
        gamma=0.99,
        policy_kwargs={"lstm_hidden_size": 256}
    )

    model.learn(total_timesteps=10000000)

    model.save(os.path.join(save_dir, "image_model_v1"))

    env.close()

if __name__ == "__main__":
    main()
