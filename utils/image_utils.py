"""
image_utils.py — Image utilities for IR->RGB SR (handles IR/RGB size mismatch)
"""

import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from PIL import Image


def create_directory(path):
    os.makedirs(path, exist_ok=True)


def denormalize(x):
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def tensor_to_pil(x):
    return to_pil_image(denormalize(x).cpu())


def save_prediction(prediction, filename):
    vutils.save_image(denormalize(prediction.squeeze(0)), filename)


def make_comparison(nir, rgb, prediction, filename):
    """
    Save side-by-side: NIR | Ground Truth | Prediction.
    NIR is upsampled to RGB resolution for visual alignment.
    """
    nir        = denormalize(nir.squeeze(0))
    rgb        = denormalize(rgb.squeeze(0))
    prediction = denormalize(prediction.squeeze(0))

    if nir.shape[0] == 1:
        nir = nir.repeat(3, 1, 1)

    # Upsample NIR to match RGB spatial size if different
    if nir.shape[-2:] != rgb.shape[-2:]:
        nir = F.interpolate(nir.unsqueeze(0), size=rgb.shape[-2:],
                            mode="bilinear", align_corners=True).squeeze(0)

    grid = torch.cat([nir, rgb, prediction], dim=2)
    vutils.save_image(grid, filename)


def save_grid(images, filename, nrow=4):
    vutils.save_image(denormalize(images), filename, nrow=nrow)