import os
import torch
import onnx
from stable_baselines3 import PPO

# 1. Stable-Baselines3 で事前学習済みモデルをロード
model_path = os.path.join("model", "RCJ_ppo_model_v1.zip")
ppo_model = PPO.load(model_path)

# 2. ポリシー部分を抽出 (PyTorch ベース)
policy = ppo_model.policy

# 3. ONNX にエクスポートするためのダミー入力生成
# 環境に応じた観測空間の形状を使用
dummy_input = torch.randn(1, *ppo_model.observation_space.shape)

# 4. PyTorch モデルを ONNX に変換
onnx_model_path = "ppo_model.onnx"
torch.onnx.export(
    policy,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=11,  # ONNX opset のバージョン指定
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"モデルが ONNX ファイルとして保存されました: {onnx_model_path}")

# 5. ONNX の動作確認 (optional)
import onnxruntime as ort

onnx_session = ort.InferenceSession(onnx_model_path)

# ONNX モデルの推論をテスト
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name
onnx_result = onnx_session.run(
    [output_name], {input_name: dummy_input.numpy()}
)

print(f"ONNX 推論結果: {onnx_result}")
