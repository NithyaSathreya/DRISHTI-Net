"""
metrics.py — PSNR, SSIM, LPIPS
Install: pip install pytorch-msssim lpips
"""

import math
import torch

try:
    from pytorch_msssim import ssim
except ImportError:
    ssim = None

try:
    import lpips as lpips_lib
    _lpips_fn = None   # lazy init — avoids loading VGG at import time
except ImportError:
    lpips_lib = None


# --------------------------------------------------------
# PSNR
# --------------------------------------------------------

def calculate_psnr(pred, target, max_val=1.0):
    """Tensors in [-1, 1]."""
    pred, target = pred.float(), target.float()
    pred   = (pred   + 1.0) * 0.5
    target = (target + 1.0) * 0.5
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return 100.0
    return 20 * math.log10(max_val) - 10 * math.log10(mse.item())


# --------------------------------------------------------
# SSIM
# --------------------------------------------------------

def calculate_ssim(pred, target):
    """Tensors in [-1, 1]."""
    if ssim is None:
        raise ImportError("pip install pytorch-msssim")
    pred   = (pred   + 1.0) * 0.5
    target = (target + 1.0) * 0.5
    return ssim(pred, target, data_range=1.0, size_average=True).item()


# --------------------------------------------------------
# LPIPS
# --------------------------------------------------------

def calculate_lpips(pred, target, device=None):
    """
    Learned Perceptual Image Patch Similarity.
    Tensors in [-1, 1] (AlexNet backbone, already expects [-1, 1]).
    Lower is better (0 = identical).
    Returns float('nan') if inputs contain NaN rather than propagating silently.
    install: pip install lpips
    """
    global _lpips_fn
    if lpips_lib is None:
        raise ImportError("pip install lpips")
    if torch.isnan(pred).any() or torch.isnan(target).any():
        return float("nan")
    if _lpips_fn is None:
        _lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False)
    dev = device or pred.device
    _lpips_fn = _lpips_fn.to(dev)
    with torch.no_grad():
        score = _lpips_fn(pred.clamp(-1.0, 1.0), target.clamp(-1.0, 1.0))
    return score.mean().item()


# --------------------------------------------------------
# Average Meter
# --------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, value, n=1):
        self.val    = value
        self.sum   += value * n
        self.count += n
        self.avg    = self.sum / self.count


# --------------------------------------------------------
# Metric Tracker  (Loss, PSNR, SSIM, LPIPS)
# --------------------------------------------------------

class MetricTracker:
    def __init__(self):
        self.loss  = AverageMeter()
        self.psnr  = AverageMeter()
        self.ssim  = AverageMeter()
        self.lpips = AverageMeter()

    def reset(self):
        for m in (self.loss, self.psnr, self.ssim, self.lpips):
            m.reset()

    def update(self, loss, psnr, ssim_val, lpips_val, batch_size=1):
        self.loss.update(loss,      batch_size)
        self.psnr.update(psnr,      batch_size)
        self.ssim.update(ssim_val,  batch_size)
        self.lpips.update(lpips_val,batch_size)

    def results(self):
        return {
            "Loss":  self.loss.avg,
            "PSNR":  self.psnr.avg,
            "SSIM":  self.ssim.avg,
            "LPIPS": self.lpips.avg,
        }