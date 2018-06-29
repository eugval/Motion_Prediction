import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt


class SimpleUNet(nn.Module):
    def __init__(self, initial_channels):
        super(SimpleUNet, self).__init__()
        self.conv1 = nn.Conv2d(initial_channels, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)

        self.deconv2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv5 = nn.Conv2d(32, 16, 3, padding=1)

        self.conv6 = nn.Conv2d(16, 8, 3, padding=1)

        self.conv7 = nn.Conv2d(8, 4, 3, padding=1)
        self.conv8 = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)

        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x2 = F.relu(x2)

        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x3 = F.relu(x3)

        x4 = self.deconv1(x3)
        x4 = torch.cat((x4, x2), 1)
        x4 = self.conv4(x4)
        x4 = F.relu(x4)

        x5 = self.deconv2(x4)
        x5 = torch.cat((x5, x1), 1)
        x5 = self.conv5(x5)
        x5 = F.relu(x5)

        x6 = self.conv6(x5)
        x6 = F.relu(x6)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        return x8


