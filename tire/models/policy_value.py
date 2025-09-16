import torch
import torch.nn as nn

class CNNPolicyValue(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(64*7*7, 512), nn.ReLU())
        self.pi = nn.Linear(512, num_actions)
        self.v = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        h = self.conv(x).view(x.size(0), -1)
        h = self.fc(h)
        return self.pi(h), self.v(h).squeeze(-1)
