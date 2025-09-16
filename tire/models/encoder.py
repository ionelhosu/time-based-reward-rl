import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomEncoder(nn.Module):
    """Frozen random CNN used for novelty embeddings."""
    def __init__(self, in_channels=4, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        self.fc = nn.Linear(64*7*7, out_dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = x / 255.0
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = F.normalize(self.fc(h), dim=-1)
        return z
