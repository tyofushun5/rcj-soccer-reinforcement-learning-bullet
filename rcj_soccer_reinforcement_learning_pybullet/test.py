import torch
from torchrl.envs import EnvBase
from torchrl.data import Unbounded, Categorical
from tensordict import TensorDict

class SimplePositionEnv(EnvBase):
    def __init__(self):
        super().__init__()
        # 状態: 実数値1次元
        self.observation_spec = Unbounded(shape=(1,))
        # 行動: 2値 (0 or 1) (0→-1, 1→+1)
        self.action_spec = Categorical(2)
        self.position = None

    def _reset(self, tensordict=None):
        self.position = torch.tensor([5.0])
        return TensorDict({"observation": self.position.clone()}, batch_size=[])

    def _step(self, action):
        action = action["action"].item()
        move = 1 if action == 1 else -1
        self.position += move
        reward = -torch.abs(self.position)
        done = torch.abs(self.position) < 1e-2 or torch.abs(self.position) > 10
        obs = TensorDict({"observation": self.position.clone()}, batch_size=[])
        return TensorDict({
            "next": obs,
            "reward": torch.tensor([reward], dtype=torch.float32),
            "done": torch.tensor([done], dtype=torch.bool),
        }, batch_size=[])

    def _set_seed(self, seed):
        torch.manual_seed(seed)

# テスト実行
if __name__ == "__main__":
    env = SimplePositionEnv()
    td = env.reset()
    for _ in range(5):
        action = TensorDict({"action": torch.tensor(1)}, batch_size=[])
        td = env.step(action)
        print(td)
