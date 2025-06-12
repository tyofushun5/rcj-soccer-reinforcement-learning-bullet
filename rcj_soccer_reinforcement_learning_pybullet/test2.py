import torch
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import Box
from tensordict import TensorDict
import pybullet as p
import pybullet_data
import numpy as np

class BulletSoccerEnv(EnvBase):
    def __init__(self, create_position, max_steps, magnitude, gui=False):
        super().__init__()
        self.max_steps = max_steps
        self.magnitude = magnitude
        self.create_position = create_position
        self.gui = gui

        # 強化学習向けの観測・行動仕様
        self.observation_spec = Box(
            low=torch.tensor([-1.0, -1.0, -1.0, -1.0]),
            high=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            shape=(4,), dtype=torch.float32
        )
        self.action_spec = Box(
            low=torch.tensor([-1.0, -1.0]),
            high=torch.tensor([1.0, 1.0]),
            shape=(2,), dtype=torch.float32
        )

        # Bullet初期化
        self._connect_bullet()
        self.reset()

    def _connect_bullet(self):
        if self.gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadSDF('stadium.sdf')

    def _reset(self, tensordict=None):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadSDF('stadium.sdf')

        # 例: ロボットとボールの位置を決定（本来はUnitクラス等でモデル生成）
        self.agent_pos = np.array([1.0, 0.5, 0.1]) + np.array(self.create_position)
        self.ball_pos = np.array([0.915, 1.8, 0.1]) + np.array(self.create_position)

        # ロボット・ボールを物理世界にspawn（モデル定義は省略）
        # 本格的にはUnit等の自作クラスでモデルを生成・管理する
        # ex: self.agent_id = p.loadURDF("robot.urdf", self.agent_pos)
        #     self.ball_id  = p.loadURDF("sphere.urdf", self.ball_pos)
        self.step_count = 0
        # 本サンプルでは物理モデル生成なし、観測値0返却
        obs = np.zeros(4, dtype=np.float32)
        return TensorDict({"observation": torch.tensor(obs)}, batch_size=[])

    def _step(self, action_td):
        self.step_count += 1
        action = action_td["action"].cpu().numpy()  # 2次元連続値

        # ここでロボット/ボールをactionに従い物理的に動かす（例: apply force/torque）
        # 本サンプルでは物理演算省略、状態遷移もダミー

        for _ in range(10):
            p.stepSimulation()

        # ダミーの観測（通常は自分で状態を更新し、計算する）
        obs = np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
        reward = np.random.randn()  # 本来は報酬関数で計算
        done = self.step_count >= self.max_steps
        # ここでgoal/outの条件を加える

        td = TensorDict(
            {
                "next": TensorDict({"observation": torch.tensor(obs)}, batch_size=[]),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done], dtype=torch.bool),
            },
            batch_size=[],
        )
        return td

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def close(self):
        p.disconnect()

# テスト用
if __name__ == "__main__":
    env = BulletSoccerEnv([0,0,0], max_steps=100, magnitude=1.0, gui=False)
    td = env.reset()
    print("reset:", td)
    for i in range(5):
        action = TensorDict({"action": torch.rand(2) * 2 - 1}, batch_size=[])
        td = env.step(action)
        print(f"step {i}:", td)
