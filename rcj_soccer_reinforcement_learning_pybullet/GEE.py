import torch
import torch.nn as nn
import torch.nn.functional as F

class GEE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 重み共有MLP
        self.entity_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, entity_states):
        # 各エンティティをエンコード（重み共有）
        local_features = self.entity_encoder(entity_states)  # (N, output_dim)
        # プーリングでまとめる（ここではmax pooling）
        global_feature, _ = torch.max(local_features, dim=0)  # (output_dim,)
        return global_feature
