import torch
import torch.nn as nn

class CNNDQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18, dueling=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        self.dueling = dueling
        if dueling:
            self.V = nn.Sequential(nn.Linear(64*7*7, 512), nn.ReLU(), nn.Linear(512, 1))
            self.A = nn.Sequential(nn.Linear(64*7*7, 512), nn.ReLU(), nn.Linear(512, num_actions))
        else:
            self.head = nn.Sequential(nn.Linear(64*7*7, 512), nn.ReLU(), nn.Linear(512, num_actions))

    def forward(self, x):
        x = x / 255.0
        h = self.conv(x).view(x.size(0), -1)
        if self.dueling:
            v = self.V(h); a = self.A(h)
            q = v + (a - a.mean(dim=1, keepdim=True))
        else:
            q = self.head(h)
        return q
