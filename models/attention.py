"""
attention.py
------------

Attention Gate for Attention U-Net.

Reference:
Attention U-Net
Oktay et al.

Author: Nithya
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention Gate

    Inputs
    ------
    x : Encoder feature map (skip connection)

    g : Decoder feature map (gating signal)

    Output
    ------
    Filtered encoder feature map
    """

    def __init__(self, skip_channels, gating_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = skip_channels // 2

        inter_channels = max(inter_channels, 1)
        #
        # Encoder projection
        #
        self.theta_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(inter_channels, affine=True)
        )
        #
        # Decoder projection
        #
        self.phi_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(inter_channels, affine=True)
        )
        #
        # Attention map
        #
        self.psi = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, g):

        #
        # Project encoder features
        #
        theta_x = self.theta_x(x)
        #
        # Project decoder features
        #
        phi_g = self.phi_g(g)
        #
        # Decoder feature may be
        # one pixel smaller because
        # of interpolation.
        #
        if theta_x.shape[-2:] != phi_g.shape[-2:]:
            phi_g = nn.functional.interpolate(phi_g, size=theta_x.shape[-2:], mode="bilinear", align_corners=True)
        #
        # Attention coefficients
        #
        attention = self.psi(theta_x + phi_g)
        #
        # Filter encoder feature
        #
        out = x * attention

        return out