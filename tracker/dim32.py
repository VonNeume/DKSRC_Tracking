import torch
import torch.nn.functional as F
from torch import nn as nn

z_dim = 512

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 64, kernel_size=4, stride=1)
        self.c1 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=4, stride=1)
        self.l1 = nn.Linear(512*20*20, z_dim)

        self.b1 = nn.BatchNorm2d(128)
        self.b2 = nn.BatchNorm2d(256)
        self.b3 = nn.BatchNorm2d(512)

    def forward(self, x):
        with torch.no_grad():
            h = F.relu(self.c0(x))
            features = F.relu(self.b1(self.c1(h)))
            h = F.relu(self.b2(self.c2(features)))
            h = F.relu(self.b3(self.c3(h)))
            encoded = self.l1(h.view(x.shape[0], -1))
        return encoded, features
