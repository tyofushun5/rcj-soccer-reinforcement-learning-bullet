import onnxruntime as ort

onnx_session = ort.InferenceSession("ppo_model.onnx")
