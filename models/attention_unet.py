"""
attention_unet.py
-----------------

Attention U-Net for IR -> RGB SR Colorization (256x256 -> 512x512)

Changes from baseline:
  - OutputHead replaced with SRHead (PixelShuffle 2x)
    => network now outputs 512x512 RGB from 256x256 IR input
  - DownBlock / UpBlock now use SEResidualBlock (channel attention)
  - Bottleneck DilatedResidualBlocks now include SE attention

Author: Nithya
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.blocks import (
    ConvBNReLU,
    MultiKernelConv,
    DownBlock,
    UpBlock,
    SRHead,
    initialize_weights
)

from models.attention import AttentionGate
from models.bottleneck import Bottleneck


class AttentionUNet(nn.Module):
    """
    Attention U-Net with 2x super-resolution output head.

    Input : (B, 1, 256, 256)  -- single-channel IR / thermal
    Output: (B, 3, 512, 512)  -- RGB colorization at 2x resolution

    Architecture
    ------------
    Stem  -> 4 x DownBlock (encoder with SE residuals)
          -> Dilated bottleneck (5 dilated SE blocks, d=1,2,4,2,1)
          -> 4 x UpBlock with AttentionGate (decoder with SE residuals)
          -> SRHead: pre-refine -> PixelShuffle(x2) -> post-refine -> Conv1x1 -> Tanh
    """

    def __init__(self, in_channels=1, out_channels=3, base_channels=64,
                 grad_checkpoint=False, use_multi_kernel=False):

        super().__init__()
        c1 = base_channels       #  64
        c2 = c1 * 2              # 128
        c3 = c2 * 2              # 256
        c4 = c3 * 2              # 512
        mk = use_multi_kernel

        ####################################################
        # Stem
        ####################################################
        stem_cls   = MultiKernelConv if mk else ConvBNReLU
        self.stem  = stem_cls(in_channels, c1)

        ####################################################
        # Encoder  (each DownBlock returns skip + downsampled)
        ####################################################
        self.enc1 = DownBlock(c1, c1, mk)   # 256->128,  skip: (B, 64,  256, 256)
        self.enc2 = DownBlock(c1, c2, mk)   # 128->64,   skip: (B, 128, 128, 128)
        self.enc3 = DownBlock(c2, c3, mk)   # 64->32,    skip: (B, 256,  64,  64)
        self.enc4 = DownBlock(c3, c4, mk)   # 32->16,    skip: (B, 512,  32,  32)

        ####################################################
        # Bottleneck  (B, 512, 16, 16)
        ####################################################
        self.bottleneck = Bottleneck(channels=c4)

        ####################################################
        # Attention Gates
        ####################################################
        self.att4 = AttentionGate(skip_channels=c4, gating_channels=c4)
        self.att3 = AttentionGate(skip_channels=c3, gating_channels=c3)
        self.att2 = AttentionGate(skip_channels=c2, gating_channels=c2)
        self.att1 = AttentionGate(skip_channels=c1, gating_channels=c1)

        ####################################################
        # Decoder  (restores to 256x256 feature space)
        ####################################################
        self.dec4 = UpBlock(in_channels=c4, skip_channels=c4, out_channels=c3, use_multi_kernel=mk)
        self.dec3 = UpBlock(in_channels=c3, skip_channels=c3, out_channels=c2, use_multi_kernel=mk)
        self.dec2 = UpBlock(in_channels=c2, skip_channels=c2, out_channels=c1, use_multi_kernel=mk)
        self.dec1 = UpBlock(in_channels=c1, skip_channels=c1, out_channels=c1, use_multi_kernel=mk)

        ####################################################
        # SR Head: 256x256 features -> 512x512 RGB
        #
        # PixelShuffle(2) rearranges (B, C*4, H, W) -> (B, C, 2H, 2W).
        # Fully learnable, no checkerboard artifacts, standard in SR.
        ####################################################
        self.head = SRHead(c1, use_multi_kernel=mk)

        self.grad_checkpoint = grad_checkpoint

        ####################################################
        # Weight Initialization
        ####################################################
        initialize_weights(self)

    def _ckpt(self, fn, *args):
        """Run fn(*args) with gradient checkpointing when enabled."""
        if self.grad_checkpoint and any(a.requires_grad for a in args
                                        if isinstance(a, torch.Tensor)):
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        ####################################################
        # Stem
        ####################################################
        x = self.stem(x)                # (B, 64,  256, 256)

        ####################################################
        # Encoder  (checkpointed — saves large activation maps)
        ####################################################
        s1, x = self._ckpt(self.enc1, x)   # s1: (B, 64,  256, 256)
        s2, x = self._ckpt(self.enc2, x)   # s2: (B, 128, 128, 128)
        s3, x = self._ckpt(self.enc3, x)   # s3: (B, 256,  64,  64)
        s4, x = self._ckpt(self.enc4, x)   # s4: (B, 512,  32,  32)

        ####################################################
        # Bottleneck
        ####################################################
        # Run bottleneck in fp32: 4 decoder bilinear upsamples + 5 residual blocks
        # amplify backward gradients ~800×; fp16 overflows (max 65504) inside the
        # bottleneck when init_scale > 4. fp32 keeps the bottleneck backward clean
        # regardless of GradScaler scale. use_reentrant=False preserves autocast ctx.
        with torch.amp.autocast('cuda', enabled=False):
            x = self._ckpt(self.bottleneck, x.float())  # (B, 512, 16, 16)

        ####################################################
        # Decoder with attention-gated skip connections
        ####################################################
        s4 = self.att4(s4, x)
        x  = self._ckpt(self.dec4, x, s4)  # (B, 256, 32, 32)

        s3 = self.att3(s3, x)
        x  = self._ckpt(self.dec3, x, s3)  # (B, 128, 64, 64)

        s2 = self.att2(s2, x)
        x  = self._ckpt(self.dec2, x, s2)  # (B, 64,  128, 128)

        s1 = self.att1(s1, x)
        x  = self._ckpt(self.dec1, x, s1)  # (B, 64,  256, 256)

        ####################################################
        # SR Head: 256x256 -> 512x512 RGB
        ####################################################
        x = self.head(x)                    # (B, 3,   512, 512)

        return x