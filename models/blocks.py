"""
blocks.py
----------

Building blocks for Attention U-Net (SR edition).

Original configuration that achieved PSNR 27.19 / SSIM 0.7206 at epoch 23
(exp_attunet_v1).

Key choices (do NOT change without evidence):
  - InstanceNorm2d(affine=True) throughout — standard for image translation;
    batch-size independent, avoids BatchNorm small-batch instability with AMP
  - MaxPool2d in DownBlock (not AvgPool)
  - bias=False in all Conv2d layers (InstanceNorm affine beta absorbs bias)
  - ReLU everywhere including after PixelShuffle

Author : Nithya M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# Conv -> BN -> ReLU
# --------------------------------------------------------

class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):

        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------------
# Multi-Kernel Fusion Conv (drop-in replacement for ConvBNReLU)
# --------------------------------------------------------

class MultiKernelConv(nn.Module):
    """
    Residual multi-scale conv — drop-in replacement for ConvBNReLU.

    Design principle:
      The parallel-branch (concat) approach forces all branches to coordinate
      from random init simultaneously. At epoch 0 no branch has any head start,
      so they collectively produce noise that stalls the combined loss.

      Solution: keep the primary 3×3 path fully intact (identical to
      ConvBNReLU) and add multi-scale context as a zero-initialized residual
      correction. At init the fuse weight is all-zero → block output is
      exactly ConvBNReLU output → loss goes down from epoch 1.
      As training progresses the 1×1 and stacked-3×3 branches learn to
      provide complementary signal that the primary path cannot.

    Architecture:
      primary   : ConvBNReLU(in, out)          — full 3×3, always active
      branch_1x1: Conv2d(in, c_aux, 1)         — pointwise / channel mix
      branch_5x5: Conv2d→IN→ReLU→Conv2d(c_aux) — stacked 3×3 (5×5 RF)
      fuse      : Conv2d(2*c_aux, out, 1)       — zero-init, residual gate
      output    : IN(ReLU(primary + fuse(cat(b1, b5))))
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Primary path — identical to ConvBNReLU, guaranteed stable baseline
        self.primary = ConvBNReLU(in_channels, out_channels)

        # Auxiliary multi-scale branches (small — ~12.5% of out_channels each)
        c_aux = max(out_channels // 8, 1)

        self.branch_1x1 = nn.Conv2d(in_channels, c_aux, kernel_size=1, bias=False)

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, c_aux, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(c_aux, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_aux, c_aux, kernel_size=3, padding=1, bias=False),
        )

        # Zero-init fusion: residual starts at exactly 0
        self.fuse = nn.Conv2d(c_aux * 2, out_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.fuse.weight)

        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        main       = self.primary(x)                             # full 3×3 path
        aux        = torch.cat([self.branch_1x1(x),
                                self.branch_5x5(x)], dim=1)     # multi-scale aux
        correction = self.fuse(aux)                              # zero at init
        return self.relu(self.norm(main + correction))


# --------------------------------------------------------
# Squeeze-and-Excitation Channel Attention
# --------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Recalibrates channel-wise feature responses by learning
    which channels to emphasize or suppress.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# --------------------------------------------------------
# Residual Block (original, kept for backward compat)
# --------------------------------------------------------

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        ) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.relu(out)


# --------------------------------------------------------
# SE Residual Block (ResidualBlock + channel attention)
# --------------------------------------------------------

class SEResidualBlock(nn.Module):
    """
    Residual block with Squeeze-and-Excitation channel attention.
    Helps the encoder/decoder focus on the most informative
    feature channels at each spatial scale.

    use_multi_kernel=True replaces the first ConvBNReLU with
    MultiKernelConv (parallel 1×1 + 3×3 + 5×5 fusion).
    """

    def __init__(self, in_channels, out_channels, use_multi_kernel=False):
        super().__init__()
        conv_cls   = MultiKernelConv if use_multi_kernel else ConvBNReLU
        self.conv1 = conv_cls(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        )
        self.se = SEBlock(out_channels)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        ) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)       # channel attention before residual add
        out += identity
        return self.relu(out)


# --------------------------------------------------------
# Encoder Block
# --------------------------------------------------------

class DownBlock(nn.Module):
    """
    MaxPool2d — DO NOT change to AvgPool.
    AvgPool divides gradient by 4 per stage (4 stages = 256x attenuation),
    causing fp16 underflow in AMP and collapse at epoch 3-6.
    MaxPool routes full gradient to the max position, no attenuation.
    """
    def __init__(self, in_channels, out_channels, use_multi_kernel=False):
        super().__init__()
        self.block = SEResidualBlock(in_channels, out_channels, use_multi_kernel)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.block(x)
        down = self.pool(feat)
        return feat, down


# --------------------------------------------------------
# Decoder Block
# --------------------------------------------------------

class UpBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, use_multi_kernel=False):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = SEResidualBlock(in_channels + skip_channels, out_channels, use_multi_kernel)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size()[-2:] != skip.size()[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# --------------------------------------------------------
# Original RGB Reconstruction Head (256x256 output)
# --------------------------------------------------------

class OutputHead(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv       = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.conv(x))


# --------------------------------------------------------
# Super-Resolution Head (2x: 256x256 -> 512x512 output)
# --------------------------------------------------------

class SRHead(nn.Module):
    """
    2x super-resolution output head using sub-pixel convolution (PixelShuffle).

    Pipeline:
      Refine (SE residual) -> PixelShuffle x2 -> Post-refine (SE residual) -> Conv 1x1 -> Tanh

    Input : (B, in_channels, H,   W)   e.g. (B, 64, 256, 256)
    Output: (B, 3,           2H, 2W)   e.g. (B, 3,  512, 512)

    ReLU after PixelShuffle — original working choice (PSNR 27.19 at epoch 23).
    The refine_pre and refine_post SEResidualBlocks handle any sub-pixel artifacts.
    """

    def __init__(self, in_channels, use_multi_kernel=False):
        super().__init__()

        # Pre-upsampling refinement
        self.refine_pre = SEResidualBlock(in_channels, in_channels, use_multi_kernel)

        # Sub-pixel convolution: expand channels by r^2=4, then shuffle
        self.pixel_shuffle = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),          # (B, in_ch*4, H, W) -> (B, in_ch, 2H, 2W)
            nn.ReLU(inplace=True),
        )

        # Post-upsampling refinement at 2x resolution
        self.refine_post = SEResidualBlock(in_channels, in_channels, use_multi_kernel)

        # Final projection to RGB
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.refine_pre(x)
        x = self.pixel_shuffle(x)
        x = self.refine_post(x)
        return self.out(x)


# --------------------------------------------------------
# Weight Initialization
# --------------------------------------------------------

def initialize_weights(model):

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
