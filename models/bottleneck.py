"""
bottleneck.py
-------------

Residual Dilated Bottleneck

Author: Nithya
"""

import torch
import torch.nn as nn


class DilatedResidualBlock(nn.Module):

    def __init__(self, channels, dilation):
        super().__init__()
        padding = dilation

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        out = self.block(x)
        out = out + x
        return self.relu(out)


class Bottleneck(nn.Module):

    """
    Residual Dilated Bottleneck
    dilation schedule d=1 -> d=2 -> d=4 -> d=2 -> d=1
    """

    def __init__(self, channels=1024):

        super().__init__()
        self.block1 = DilatedResidualBlock(channels, dilation=1)
        self.block2 = DilatedResidualBlock(channels, dilation=2)
        self.block3 = DilatedResidualBlock(channels, dilation=4)
        self.block4 = DilatedResidualBlock(channels, dilation=2)
        self.block5 = DilatedResidualBlock(channels, dilation=1)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x