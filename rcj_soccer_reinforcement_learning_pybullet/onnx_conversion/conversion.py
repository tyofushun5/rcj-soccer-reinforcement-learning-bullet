import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
import os

class ONNXWrappedPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, h_pi, c_pi, h_vf, c_vf, episode_starts):
        # actor/criticのLSTM状態をタプルにまとめる
        lstm_states = ((h_pi, c_pi), (h_vf, c_vf))

        # RecurrentPPO のポリシーでは forward(obs, lstm_states, episode_starts, deterministic=False) の形
        # 戻り値は (actions, new_lstm_states, その他) ですが、
        # コード上は (action, new_lstm_states) だったりしますので、必要に応じて修正してください
        actions, new_lstm_states = self.policy.forward(
            obs=obs,
            lstm_states=lstm_states,
            episode_starts=episode_starts,
            deterministic=False  # 必要に応じて
        )

        # 返ってきた new_lstm_states は ((new_h_pi, new_c_pi), (new_h_vf, new_c_vf)) になっているはず
        (new_h_pi, new_c_pi), (new_h_vf, new_c_vf) = new_lstm_states

        # ONNX 出力用に hidden/cell をまとめて返す
        return actions, new_h_pi, new_c_pi, new_h_vf, new_c_vf

# ここから下は同様
model_path = os.path.join("..", "model", "default_model", "default_model_v1.zip")
model = RecurrentPPO.load(model_path)
wrapped_policy = ONNXWrappedPolicy(model.policy)

obs = torch.rand((1, 4), dtype=torch.float32)
h_pi = torch.zeros((1, 1, 256))
c_pi = torch.zeros((1, 1, 256))
h_vf = torch.zeros((1, 1, 256))
c_vf = torch.zeros((1, 1, 256))

episode_starts = torch.zeros((1,), dtype=torch.float32)  # 全部続き扱い、先頭ではない例

# ONNX 出力
torch.onnx.export(
    wrapped_policy,
    (obs, h_pi, c_pi, h_vf, c_vf, episode_starts),
    "default_model_v1.onnx",
    input_names=["obs", "h_pi", "c_pi", "h_vf", "c_vf", "episode_starts"],
    output_names=["action", "new_h_pi", "new_c_pi", "new_h_vf", "new_c_vf"],
    dynamic_axes={
        "obs": {0: "batch"},
        "episode_starts": {0: "batch"}
    },
    opset_version=11
)