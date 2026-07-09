# -*- coding: utf-8 -*-
"""
model_factory.py
----------------

Config-driven model builder.

Reads model.type from config dict / object and returns the appropriate
model. Training loop code needs no if/else on model type -- it just calls
build_model(cfg) and the rest is handled here.

Supported types
---------------
  "attention_unet"  : plain AttentionUNet (single optimizer, simple loop)
  "gan_unet"        : GANUNet (two optimizers, alternating D/G steps)

Config schema (dict or any object with attribute access)
---------------------------------------------------------
model:
  type            : "attention_unet" | "gan_unet"
  in_channels     : 1           # IR (single channel thermal)
  out_channels    : 3           # RGB
  base_channels   : 64          # encoder/decoder width
  gan:
    d_base_channels : 64        # discriminator width
    adv_weight      : 0.01      # adversarial vs reconstruction tradeoff
    lr              : 0.0002    # Adam learning rate for both G and D
    betas           : [0.5, 0.999]

Example usage
-------------
  # with a plain dict
  cfg = {
      "model": {
          "type": "gan_unet",
          "in_channels": 1,
          "out_channels": 3,
          "base_channels": 64,
          "gan": {
              "d_base_channels": 64,
              "adv_weight": 0.01,
              "lr": 2e-4,
              "betas": [0.5, 0.999],
          }
      }
  }
  model = build_model(cfg)

  # Training loop (GAN branch)
  if is_gan(cfg):
      opt_G, opt_D = model.build_optimizers(...)
      for ir, rgb in dataloader:
          d_loss            = model.discriminator_step(ir, rgb)
          opt_D.zero_grad(); d_loss.backward(); opt_D.step()

          g_loss, losses    = model.generator_step(ir, rgb, recon_loss_fn)
          opt_G.zero_grad(); g_loss.backward(); opt_G.step()
  else:
      opt = torch.optim.Adam(model.parameters(), ...)
      for ir, rgb in dataloader:
          pred              = model(ir)
          loss, losses      = recon_loss_fn(pred, rgb)
          opt.zero_grad(); loss.backward(); opt.step()

Author: Nithya
"""

import torch
from models.attention_unet import AttentionUNet
from models.gan_unet        import GANUNet


# ----------------------------------------------------------
# Config accessor (supports both dict and object notation)
# ----------------------------------------------------------

def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ----------------------------------------------------------
# Public helpers
# ----------------------------------------------------------

def is_gan(cfg) -> bool:
    """Returns True if config requests a GAN model."""
    model_cfg = _get(cfg, "model", {})
    return _get(model_cfg, "type", "attention_unet") == "gan_unet"


def build_model(cfg) -> torch.nn.Module:
    """
    Build and return the model specified in cfg.

    Parameters
    ----------
    cfg : dict or config object with a "model" key/attribute.

    Returns
    -------
    AttentionUNet or GANUNet instance (both have identical forward() signature).
    """
    model_cfg = _get(cfg, "model", {})
    model_type    = _get(model_cfg, "type",          "attention_unet")
    in_channels   = _get(model_cfg, "in_channels",   1)
    out_channels  = _get(model_cfg, "out_channels",  3)
    base_channels = _get(model_cfg, "base_channels", 64)

    grad_checkpoint  = _get(model_cfg, "grad_checkpoint",  False)
    use_multi_kernel = _get(model_cfg, "multi_kernel",     False)

    if model_type == "attention_unet":
        return AttentionUNet(
            in_channels      = in_channels,
            out_channels     = out_channels,
            base_channels    = base_channels,
            grad_checkpoint  = grad_checkpoint,
            use_multi_kernel = use_multi_kernel,
        )

    elif model_type == "gan_unet":
        gan_cfg         = _get(model_cfg, "gan", {})
        d_base_channels = _get(gan_cfg, "d_base_channels", 64)
        adv_weight      = _get(gan_cfg, "adv_weight",      0.001)
        fm_weight       = _get(gan_cfg, "fm_weight",       1.0)
        num_scales      = _get(gan_cfg, "num_scales",      3)

        return GANUNet(
            in_channels     = in_channels,
            out_channels    = out_channels,
            base_channels   = base_channels,
            d_base_channels = d_base_channels,
            num_scales      = num_scales,
            adv_weight      = adv_weight,
            fm_weight       = fm_weight,
            grad_checkpoint = grad_checkpoint,
        )

    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose 'attention_unet' or 'gan_unet'."
        )


def _make_param_groups(model, weight_decay):
    """
    Split parameters into two groups:

      decay    — Conv2d/Linear weights (ndim > 1, no 'norm' in name, no '.bias')
                 weight_decay applied here.

      no_decay — InstanceNorm2d/BatchNorm affine (gamma/beta, ndim==1),
                 any bias, and anything with 'norm' in its name.
                 weight_decay=0.0

    Applies to both Adam and AdamW:
      - Adam:  L2 reg is coupled to adaptive scaling — norm params with large
               wd get pulled to zero, collapsing normalisation (plateau ~epoch 10).
      - AdamW: wd is decoupled (applied directly to weights, not gradients),
               so it is safer in principle, but excluding norm/bias is still
               standard practice and costs nothing.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or 'norm' in name or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)

    n_decay    = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    print(f"[Optimizer] decay={n_decay/1e6:.2f}M  "
          f"no-decay(norm+bias)={n_no_decay/1e6:.2f}M  "
          f"weight_decay={weight_decay}")

    return [
        {'params': decay,    'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


_OPTIMIZERS = {
    'adam':  torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}


def build_optimizers(model, cfg):
    """
    Build optimizer(s) appropriate for the model type.

    Optimizer is selected via cfg["training"]["optimizer"]:
      "adam"  (default) — Adam with L2 reg on conv weights only (norm/bias excluded)
      "adamw"           — AdamW with decoupled weight decay (norm/bias still excluded)

    Returns
    -------
    For attention_unet : (opt,)              -- single-element tuple
    For gan_unet       : (opt_G, opt_D)      -- two optimizers
    """
    model_cfg = _get(cfg, "model", {})
    model_type = _get(model_cfg, "type", "attention_unet")

    if model_type == "attention_unet":
        train_cfg    = _get(cfg, "training", {})
        lr           = _get(train_cfg, "lr",           5e-4)
        betas        = tuple(_get(train_cfg, "betas",  [0.9, 0.999]))
        weight_decay = _get(train_cfg, "weight_decay", 1e-2)
        opt_name     = _get(train_cfg, "optimizer",    "adam").lower()

        if opt_name not in _OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{opt_name}'. Choose 'adam' or 'adamw'.")

        opt_cls      = _OPTIMIZERS[opt_name]
        param_groups = _make_param_groups(model, weight_decay)
        print(f"[Optimizer] type={opt_name.upper()}  lr={lr}  betas={betas}")
        opt = opt_cls(param_groups, lr=lr, betas=betas)
        return (opt,)

    elif model_type == "gan_unet":
        gan_cfg = _get(model_cfg, "gan", {})
        lr      = _get(gan_cfg, "lr",    2e-4)
        betas   = tuple(_get(gan_cfg, "betas", [0.5, 0.999]))
        return model.build_optimizers(lr=lr, betas=betas)  # returns (opt_G, opt_D)

    else:
        raise ValueError(f"Unknown model type '{model_type}'.")


# ----------------------------------------------------------
# Training step dispatcher
# ----------------------------------------------------------

def training_step(model, ir, real_rgb, recon_loss_fn, optimizers, cfg):
    """
    Unified training step that works for both model types.
    Handles D/G alternation internally for GAN, single step for plain U-Net.

    Parameters
    ----------
    model         : output of build_model()
    ir            : (B, 1, 256, 256) IR input tensor
    real_rgb      : (B, 3, 512, 512) ground truth RGB tensor
    recon_loss_fn : e.g. CombinedLoss_v3()
    optimizers    : output of build_optimizers() -- tuple of 1 or 2 optimizers
    cfg           : same config dict used to build the model

    Returns
    -------
    loss_dict : dict with named loss values for logging
    """
    if is_gan(cfg):
        opt_G, opt_D = optimizers

        # --- Discriminator step ---
        d_loss = model.discriminator_step(ir, real_rgb)
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # --- Generator step ---
        g_loss, loss_dict = model.generator_step(ir, real_rgb, recon_loss_fn)
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        loss_dict["D_loss"] = d_loss.item()

    else:
        (opt,) = optimizers
        pred   = model(ir)
        loss, loss_dict = recon_loss_fn(pred, real_rgb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss_dict