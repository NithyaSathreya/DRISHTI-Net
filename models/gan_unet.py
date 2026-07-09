# -*- coding: utf-8 -*-
"""
gan_unet.py
-----------

Pix2PixHD-lite: AttentionUNet generator + Multi-scale PatchGAN discriminator.

  Generator     : AttentionUNet (256×256 IR → 512×512 RGB)  [unchanged]
  Discriminator : MultiScaleDiscriminator — 3 PatchGANs at 512, 256, 128
  Adversarial   : LSGAN (MSE) averaged across scales
  Feature match : L1 on D's intermediate features (weight=10, Pix2PixHD)
  Recon loss    : CombinedLoss_v3 (passed in from training loop)

Key improvement over plain Pix2Pix
-------------------------------------
Feature matching loss replaces the need for a perfectly balanced D/G.
G learns to match D's internal feature statistics for real vs fake at
multiple spatial scales — a stable, information-rich signal even when D
temporarily dominates or lags.

Training pattern
----------------
  d_loss            = model.discriminator_step(ir, real_rgb)
  g_loss, loss_dict = model.generator_step(ir, real_rgb, recon_loss_fn)

Both optimizers are built with model.build_optimizers().
D should use a lower LR than G (e.g. 5e-5 vs 2e-4).

Author: Nithya
"""

import torch
import torch.nn as nn

from models.attention_unet import AttentionUNet
from models.discriminator  import MultiScaleDiscriminator, LSGANLoss
from losses.losses         import FeatureMatchingLoss


class GANUNet(nn.Module):
    """
    Pix2PixHD-lite model.

    Parameters
    ----------
    in_channels     : IR input channels (1)
    out_channels    : RGB output channels (3)
    base_channels   : generator base width (64)
    d_base_channels : discriminator base width (64)
    num_scales      : number of discriminator scales (3)
    adv_weight      : adversarial loss weight (0.01)
    fm_weight       : feature matching loss weight (10.0)
    grad_checkpoint : gradient checkpointing in generator
    """

    def __init__(
        self,
        in_channels     = 1,
        out_channels    = 3,
        base_channels   = 64,
        d_base_channels = 64,
        num_scales      = 3,
        adv_weight      = 0.1,
        fm_weight       = 10.0,
        grad_checkpoint = False,
    ):
        super().__init__()

        self.adv_weight = adv_weight

        self.generator = AttentionUNet(
            in_channels     = in_channels,
            out_channels    = out_channels,
            base_channels   = base_channels,
            grad_checkpoint = grad_checkpoint,
        )

        # Conditional D input: cat(IR, RGB) = in_channels + out_channels
        self.discriminator = MultiScaleDiscriminator(
            in_channels  = in_channels + out_channels,
            base_channels= d_base_channels,
            num_scales   = num_scales,
        )

        self.gan_loss = LSGANLoss()
        self.fm_loss  = FeatureMatchingLoss(weight=fm_weight)

    # ----------------------------------------------------------
    # Forward (inference — generator only)
    # ----------------------------------------------------------

    def forward(self, ir):
        return self.generator(ir)

    def generator_parameters(self):
        return self.generator.parameters()

    def discriminator_parameters(self):
        return self.discriminator.parameters()

    # ----------------------------------------------------------
    # Discriminator training step
    # ----------------------------------------------------------

    def discriminator_step(self, ir, real_rgb):
        """
        Compute D loss averaged across all scales.
        Backprop and opt_D.step() happen in the training loop.
        """
        with torch.no_grad():
            fake_rgb = self.generator(ir)

        real_results = self.discriminator(ir, real_rgb.detach())
        fake_results = self.discriminator(ir, fake_rgb.detach())

        d_loss = 0.0
        for (real_pred, _), (fake_pred, _) in zip(real_results, fake_results):
            d_loss = d_loss + self.gan_loss.discriminator_loss(real_pred, fake_pred)
        return d_loss / len(real_results)

    # ----------------------------------------------------------
    # Generator training step
    # ----------------------------------------------------------

    def generator_step(self, ir, real_rgb, recon_loss_fn):
        """
        G loss = reconstruction + adv_weight * adversarial + feature_matching.

        Feature matching dominates early training (weight=10) — it gives G
        a stable gradient even when D is imperfect. Adversarial (weight=0.01)
        adds texture sharpness once G is producing plausible outputs.
        """
        fake_rgb = self.generator(ir)

        # Real features: no_grad — we only need feature values, not gradients through D.
        # Gradients should only flow through the fake path (fake → D → G).
        # Without this, D's parameters accumulate spurious gradients from the real pass
        # during G's backward, corrupting D's next update.
        with torch.no_grad():
            real_results = self.discriminator(ir, real_rgb)

        # Freeze D so G's backward doesn't populate D's .grad
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        fake_results = self.discriminator(ir, fake_rgb)
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        fake_results = self.discriminator(ir, fake_rgb)

        # Adversarial loss: fool D across all scales
        adv_loss = 0.0
        for (fake_pred, _) in fake_results:
            adv_loss = adv_loss + self.gan_loss.generator_loss(fake_pred)
        adv_loss = adv_loss / len(fake_results)

        # Feature matching loss: match D's internal features for real vs fake
        real_feats = [feats for _, feats in real_results]
        fake_feats = [feats for _, feats in fake_results]
        fm_loss = self.fm_loss(real_feats, fake_feats)

        # Reconstruction loss
        with torch.amp.autocast('cuda', enabled=False):
            #recon_loss, loss_dict = recon_loss_fn(fake_rgb, real_rgb)
            recon_loss, loss_dict = recon_loss_fn(fake_rgb.float(), real_rgb.float())

        g_loss = recon_loss + self.adv_weight * adv_loss + fm_loss

        loss_dict["Adversarial"] = adv_loss.item()
        loss_dict["FeatMatch"]   = fm_loss.item()
        loss_dict["Total"]       = g_loss.item()

        return g_loss, loss_dict

    # ----------------------------------------------------------
    # Build optimizers
    # ----------------------------------------------------------

    def build_optimizers(self, lr=2e-4, betas=(0.5, 0.999)):
        """
        Returns (opt_G, opt_D).
        G uses betas=(0.9, 0.999) — stable for both reconstruction warmup and GAN training.
        D uses betas=(0.5, 0.999) — low momentum prevents D from becoming overconfident.
        D LR should be overridden to ~5e-5 in the training loop via args.d_lr.
        """
        opt_G = torch.optim.AdamW(self.generator_parameters(),     lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        opt_D = torch.optim.AdamW(self.discriminator_parameters(), lr=lr, betas=betas,  weight_decay=1e-4)
        return opt_G, opt_D