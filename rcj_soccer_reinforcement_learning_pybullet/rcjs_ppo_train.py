import os

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from rcj_soccer_reinforcement_learning_pybullet.environment.goal_environment import GoalEnvironment

save_dir = "model"
os.makedirs(save_dir, exist_ok=True)

def main():
    # 環境の作成
    env = GoalEnvironment(create_position=[4.0, 0.0, 0.0],
                          max_steps=5000,
                          magnitude=10.0,
                          gui=True)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        device="cuda",
        verbose=1,
        n_epochs=10,
        n_steps=128,
        batch_size=128,
        gamma=0.99,
        policy_kwargs={"lstm_hidden_size": 256}
    )

    model.learn(total_timesteps=5000000)

    model.save(os.path.join(save_dir, "RCJ_ppo_model_v2"))

    env.close()

if __name__ == "__main__":
    main()
