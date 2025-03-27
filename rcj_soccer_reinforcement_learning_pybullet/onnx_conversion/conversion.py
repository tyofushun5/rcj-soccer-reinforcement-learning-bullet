import os
import torch
import onnx
from sb3_contrib import RecurrentPPO

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
model_path = os.path.join(parent_dir, "model", "goal_model", "dispersion_goal_model_v1")
model = RecurrentPPO.load(model_path)
policy = model.policy

# ダミー入力の準備
batch_size = 1
seq_len = 1  # ONNXでは1ステップ分として扱う

obs_shape = model.observation_space.shape
dummy_obs = torch.randn(seq_len, batch_size, *obs_shape)

if hasattr(policy, "lstm"):
    lstm_module = policy.lstm
elif hasattr(policy, "recurrent_net"):
    lstm_module = policy.recurrent_net
else:
    raise AttributeError("LSTMモジュールが見つかりません。policy.lstm または policy.recurrent_net を確認してください。")

num_layers = lstm_module.num_layers
hidden_size = lstm_module.hidden_size
device = next(policy.parameters()).device

h0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
c0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
dummy_hidden = torch.cat([h0, c0], dim=0)  # ONNXエクスポート用に連結

# episode_starts（マスク）の準備
dummy_masks = torch.ones(seq_len, batch_size)

# ONNXエクスポート用のラッパー関数
class RecurrentPolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, lstm_state, mask):
        # 連結された lstm_state を (h, c) に分割
        hidden_dim = lstm_state.size(0) // 2
        h, c = lstm_state[:hidden_dim], lstm_state[hidden_dim:]
        lstm_states = (h, c)

        # ポリシー実行
        action, _, _, _ = self.policy(obs, lstm_states, mask)
        return action

# ラップしてONNX形式に変換
wrapper = RecurrentPolicyWrapper(policy)
onnx_model_path = "recurrent_model.onnx"

# 入力の準備（ラッパーの引数に合わせて順番を設定）
torch.onnx.export(
    wrapper,
    (dummy_obs, dummy_hidden, dummy_masks),
    onnx_model_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["obs", "lstm_state", "mask"],
    output_names=["action"],
    dynamic_axes={
        "obs": {0: "seq_len", 1: "batch_size"},
        "lstm_state": {1: "batch_size"},
        "mask": {0: "seq_len", 1: "batch_size"},
        "action": {0: "seq_len", 1: "batch_size"},
    },
)
