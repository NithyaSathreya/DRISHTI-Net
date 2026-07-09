# -*- coding: utf-8 -*-
"""
discriminator.py
----------------

Pix2PixHD-lite multi-scale discriminator for IR -> RGB SR colorization.

Changes from plain Pix2Pix:
  - PatchGANDiscriminator now returns (pred, intermediate_features)
    so the training loop can compute feature matching loss.
  - MultiScaleDiscriminator runs 3 PatchGANs at 512, 256, 128 resolution.
    The discriminator sees the output at multiple spatial scales, catching
    both coarse colour errors (low-res D) and fine texture errors (full-res D).
  - InstanceNorm replaces BatchNorm (batch-size independent).
  - Spectral norm on every conv (prevents D from dominating G).
  - One-sided label smoothing: real_label=0.9 (LSGANLoss).

Author: Nithya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------
# Discriminator conv block: SpectralNorm → InstanceNorm → LeakyReLU
# ----------------------------------------------------------

class DiscBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, use_norm=False):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not use_norm)
            )
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# class DiscBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, stride=2, use_norm=True):
#         super().__init__()
#         layers = [
#             nn.utils.spectral_norm(
#                 nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=True)
#             )
#         ]
#         if use_norm:
#             layers.append(nn.InstanceNorm2d(out_ch, affine=True))
#         layers.append(nn.LeakyReLU(0.2, inplace=True))
#         self.block = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.block(x)


# ----------------------------------------------------------
# Single-scale PatchGAN — returns (pred, [feat0..feat3])
# ----------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """
    70×70 conditional PatchGAN discriminator.
    Returns both the patch-score map and intermediate feature maps
    so the caller can compute feature matching loss.

    Input : cat(IR_upsampled, RGB)  →  (B, 4, H, W)
    Output: pred  (B, 1, H', W')
            feats [f0, f1, f2, f3]  — one per DiscBlock
    """

    def __init__(self, in_channels=4, base_channels=64):
        super().__init__()
        bc = base_channels
        # Stored as ModuleList so we can extract intermediate features
        self.layers = nn.ModuleList([
            DiscBlock(in_channels, bc,     stride=2, use_norm=False),  # 0: no norm on first
            DiscBlock(bc,          bc * 2, stride=2, use_norm=True),   # 1
            DiscBlock(bc * 2,      bc * 4, stride=2, use_norm=True),   # 2
            DiscBlock(bc * 4,      bc * 8, stride=1, use_norm=True),   # 3: stride 1
        ])
        self.out = nn.utils.spectral_norm(
            nn.Conv2d(bc * 8, 1, kernel_size=4, stride=1, padding=1, bias=True)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, ir, rgb):
        if ir.shape[-2:] != rgb.shape[-2:]:
            ir = F.interpolate(ir, size=rgb.shape[-2:],
                               mode="bilinear", align_corners=True)
        x = torch.cat([ir, rgb], dim=1)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        pred = self.out(x)
        return pred, feats   # (B,1,H',W'),  [f0,f1,f2,f3]


# ----------------------------------------------------------
# Multi-scale Discriminator  (Pix2PixHD-style)
# ----------------------------------------------------------

class MultiScaleDiscriminator(nn.Module):
    """
    Runs num_scales independent PatchGANs on progressively
    downsampled versions of the input:
        D0 → full resolution  (e.g. 512×512)
        D1 → half resolution  (256×256)
        D2 → quarter resolution (128×128)

    Why multi-scale?
      D0 evaluates fine texture and local sharpness.
      D1/D2 evaluate global structure and colour consistency.
      Together they give the generator richer gradient signal than a
      single fixed-scale discriminator.

    forward() returns a list of (pred, feats) — one per scale — so the
    training loop can sum adversarial losses and compute feature matching.
    """

    def __init__(self, in_channels=4, base_channels=64, num_scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(in_channels, base_channels)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1,
                                       count_include_pad=False)

    def forward(self, ir, rgb):
        """
        Returns
        -------
        results : list of (pred, feats), one entry per scale.
                  pred  : (B, 1, H', W')
                  feats : [f0, f1, f2, f3]
        """
        results = []
        x_ir, x_rgb = ir, rgb
        for D in self.discriminators:
            results.append(D(x_ir, x_rgb))
            x_ir  = self.downsample(x_ir)
            x_rgb = self.downsample(x_rgb)
        return results


# ----------------------------------------------------------
# LSGAN Loss  (one-sided label smoothing: real=0.9)
# ----------------------------------------------------------

class LSGANLoss(nn.Module):
    """
    Least-Squares GAN loss (Mao et al. 2017).
    real_label=0.9 (one-sided smoothing) prevents D from becoming
    overconfident on real samples.
    """

    def __init__(self, real_label=0.9, fake_label=0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.MSELoss()

    def _label(self, pred, is_real):
        val = self.real_label if is_real else self.fake_label
        return torch.full_like(pred, val)

    def discriminator_loss(self, real_pred, fake_pred):
        loss_real = self.loss(real_pred, self._label(real_pred, True))
        loss_fake = self.loss(fake_pred, self._label(fake_pred, False))
        return (loss_real + loss_fake) * 0.5

    def generator_loss(self, fake_pred):
        return self.loss(fake_pred, self._label(fake_pred, True))