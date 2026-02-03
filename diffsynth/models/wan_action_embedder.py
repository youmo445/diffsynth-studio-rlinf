import torch
import torch.nn as nn
import numpy as np
class ActionEmbedder(nn.Module):
    def __init__(self, action_dim=8, embed_dim=1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, action):  # action: [T, 8] or [B, T, 8]
        if action.ndim == 2:
            action = action.unsqueeze(0)  # [1, T, 8]
        B, T, D = action.shape
        action = action.reshape(B * T, D)
        emb = self.mlp(action)  # [B*T, embed_dim]
        emb = emb.view(B, T, -1)  # [B, T, embed_dim]
        return emb