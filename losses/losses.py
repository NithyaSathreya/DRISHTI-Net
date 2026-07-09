# -*- coding: utf-8 -*-
"""
losses.py
----------

Loss functions for IR -> RGB SR Colorization (256x256 -> 512x512)

v1  : L1 + SSIM
v2  : EdgeAwareCharbonnier + SSIM + SmoothL1          (baseline used)
v3  : MultiScaleEdgeCharbonnier + FFT + LabColor + SSIM + SmoothL1  (SR upgrade)

Author: Nithya  |  v3 upgrade: Sathish
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import SSIM
    from pytorch_msssim import ms_ssim
except ImportError:
    SSIM = None

from utils.metrics import calculate_psnr


# ----------------------------------------------------------
# Edge-Aware Charbonnier Loss  (unchanged from v2)
# ----------------------------------------------------------

class EdgeAwareCharbonnierLoss(nn.Module):
    """
    Charbonnier (smooth L1) loss weighted by the target's Sobel gradient
    magnitude. Edges receive higher penalty, encouraging sharper output.
    """

    def __init__(self, edge_weight=1.0, eps=1e-6):
        super().__init__()
        self.eps         = eps
        self.edge_weight = edge_weight

        kernel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
        ).view(1, 3, 3, 3)

        kernel_y = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
        ).view(1, 3, 3, 3)

        self.kernel_x = nn.Parameter(kernel_x, requires_grad=False)
        self.kernel_y = nn.Parameter(kernel_y, requires_grad=False)

    def forward(self, pred, target):
        grad_x   = F.conv2d(target, self.kernel_x.to(target.device), padding=1)
        grad_y   = F.conv2d(target, self.kernel_y.to(target.device), padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)
        weights  = 1.0 + self.edge_weight * grad_mag
        diff     = pred - target
        charb    = torch.sqrt(diff ** 2 + self.eps ** 2)
        return torch.mean(weights * charb)


# ----------------------------------------------------------
# NEW: Multi-Scale Edge-Aware Charbonnier Loss
# ----------------------------------------------------------

class MultiScaleEdgeCharbonnierLoss(nn.Module):
    """
    Applies EdgeAwareCharbonnierLoss at multiple spatial scales
    (full, 1/2, 1/4 resolution) using average-pooling downsampling.

    Why it helps for SR:
      SR outputs contain details at multiple spatial frequencies.
      Single-scale edge loss only enforces sharpness at the output
      resolution. Multi-scale supervision also ensures correct structure
      at coarser levels, preventing the network trading coarse accuracy
      for fine-grain sharpness.

    Finer scales are weighted more heavily (scale factor = 1/2^s
    prevents coarse losses from dominating).
    """

    def __init__(self, scales=3, edge_weight=1.0, eps=1e-6):
        super().__init__()
        self.scales    = scales
        self.edge_loss = EdgeAwareCharbonnierLoss(edge_weight, eps)

    def forward(self, pred, target):
        total = self.edge_loss(pred, target)           # full resolution
        for s in range(1, self.scales):
            factor   = 2 ** s
            p_down   = F.avg_pool2d(pred,   factor, factor)
            t_down   = F.avg_pool2d(target, factor, factor)
            total    = total + (1.0 / factor) * self.edge_loss(p_down, t_down)
        return total / self.scales


# ----------------------------------------------------------
# NEW: FFT Frequency Loss
# ----------------------------------------------------------

class FFTFrequencyLoss(nn.Module):
    """
    L1 loss on the magnitude of the 2D FFT spectrum.

    Why it helps for SR:
      Pixel-space losses (L1, Charbonnier) are dominated by low-frequency
      components because there are fewer high-frequency pixels. The network
      learns to produce blurry outputs that minimize average pixel error.
      The FFT loss treats all frequency bands equally: errors in high-frequency
      components (texture, edges, fine detail) are penalised just as much as
      low-frequency errors. This is one of the most impactful additions for
      super-resolution tasks.

    Uses torch.fft.rfft2 (real FFT, no complex output overhead).
    norm="ortho" keeps magnitudes scale-independent.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # rfft2 does not support fp16 (ComplexHalf is experimental and produces NaN).
        # Upcast to float32 for the FFT; result is returned as float32 scalar,
        # which is fine -- GradScaler handles the gradient scaling.
        pred_f   = pred.float()
        target_f = target.float()
        pred_mag   = torch.abs(torch.fft.rfft2(pred_f,   norm="ortho"))
        target_mag = torch.abs(torch.fft.rfft2(target_f, norm="ortho"))
        return F.l1_loss(pred_mag, target_mag)


# ----------------------------------------------------------
# NEW: Lab Color Loss
# ----------------------------------------------------------

class LabColorLoss(nn.Module):
    """
    Computes L1 loss in CIE Lab color space with upweighted chrominance.

    Why it helps for colorization:
      RGB L1 loss treats all three channels equally, but human vision
      is far more sensitive to luminance (L*) error than chrominance
      (a*, b*) error. More critically, for thermal-to-RGB colorization
      the network tends to produce desaturated / grey images because
      predicting mid-range grey minimises RGB L1 in expectation.
      Computing loss in Lab and weighting a*/b* channels 2x over L*
      explicitly penalises colour desaturation, pushing the network to
      commit to a predicted hue.

    Input RGB assumed in [-1, 1] range (network Tanh output).
    Conversion: sRGB -> linear RGB -> CIE XYZ (D65) -> CIE Lab.
    """

    def _rgb_to_lab(self, rgb_tanh):
        # Run entirely in float32 -- power ops overflow in fp16.
        rgb_tanh = rgb_tanh.float()

        # [-1, 1] -> [0, 1]
        rgb = (rgb_tanh + 1.0) * 0.5
        rgb = rgb.clamp(0.0, 1.0)

        # sRGB gamma -> linear.
        # torch.where computes gradients for BOTH branches, so the **2.4
        # gradient bleeds into the linear branch and vice-versa.
        # Use separate masked computes to avoid gradient NaN.
        mask      = rgb > 0.04045
        rgb_lin   = (rgb + 0.055) / 1.055
        # clamp before power to keep gradient finite (grad of x**2.4 at x=0 is 0,
        # but floating imprecision can produce tiny negatives after div).
        rgb_gamma = torch.exp(2.4 * torch.log(rgb_lin.clamp(min=1e-6)))
        rgb_low   = rgb / 12.92
        rgb = rgb_gamma * mask + rgb_low * (~mask)

        # Linear RGB -> XYZ (D65)
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

        # XYZ -> f(t) nonlinearity (CIE standard)
        eps   = 216.0 / 24389.0   # ~0.00886
        kappa = 24389.0 / 27.0    # ~903.3
        xn, yn, zn = 0.95047, 1.00000, 1.08883

        def f(t, n):
            t_n    = t / n
            mask_c = t_n > eps
            # Cube-root branch: clamp to 1e-4 so gradient (1/3)*x^(-2/3)
            # stays below ~1550 instead of ~21M at 1e-8.
            # Evaluate both branches without torch.where to avoid NaN gradient bleed.
            cubic  = torch.exp(torch.log(t_n.clamp(min=1e-4)) / 3.0)
            linear = (kappa * t_n + 16.0) / 116.0
            return cubic * mask_c + linear * (~mask_c)

        fx   = f(X, xn)
        fy   = f(Y, yn)
        fz   = f(Z, zn)

        L    = 116.0 * fy - 16.0
        a    = 500.0 * (fx - fy)
        b_ch = 200.0 * (fy - fz)

        return torch.cat([L, a, b_ch], dim=1)   # (B, 3, H, W)

    def forward(self, pred, target):
        pred_lab   = self._rgb_to_lab(pred)
        target_lab = self._rgb_to_lab(target)

        #L_loss  = F.l1_loss(pred_lab[:, 0:1], target_lab[:, 0:1])
        #ab_loss = F.l1_loss(pred_lab[:, 1:3], target_lab[:, 1:3])
        L_loss  = F.l1_loss(pred_lab[:, 0:1], target_lab[:, 0:1]) / 100.0   # L ∈ [0,100]
        ab_loss = F.l1_loss(pred_lab[:, 1:3], target_lab[:, 1:3]) / 128.0   # a,b ∈ [-128,127]


        # Upweight chrominance 2x to fight desaturation
        return L_loss + 2.0 * ab_loss


# ----------------------------------------------------------
# Feature Matching Loss  (Pix2PixHD)
# ----------------------------------------------------------

class FeatureMatchingLoss(nn.Module):
    """
    Pix2PixHD feature matching loss (Wang et al., 2018).

    Instead of fooling D at the output level (adversarial loss), G is
    trained to match the intermediate feature activations that D produces
    for real images. This is far more stable than pixel-level adversarial
    loss, especially with small datasets, because:
      - The gradient signal comes from D's learned mid-level features
        (edges, textures, colour blobs) rather than raw patch scores.
      - D does not need to be in perfect equilibrium for G to get a
        meaningful learning signal.
      - Feature space is smoother than output space: small changes in G
        produce small changes in the loss.

    Usage
    -----
    real_results = discriminator(ir, real_rgb)   # list of (pred, feats)
    fake_results = discriminator(ir, fake_rgb)
    real_feats = [feats for _, feats in real_results]
    fake_feats = [feats for _, feats in fake_results]
    loss = FeatureMatchingLoss()(real_feats, fake_feats)

    weight=10.0 per the Pix2PixHD paper.
    """

    def __init__(self, weight=10.0):
        super().__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, real_feats_per_scale, fake_feats_per_scale):
        """
        Parameters
        ----------
        real_feats_per_scale : list of lists — one inner list per D scale,
                               each inner list contains feature tensors
                               [f0, f1, f2, f3].
        fake_feats_per_scale : same structure for generated images.

        Returns
        -------
        Weighted average L1 across all scales and all feature layers.
        """
        loss = 0.0
        n    = 0
        for real_feats, fake_feats in zip(real_feats_per_scale,
                                          fake_feats_per_scale):
            for rf, ff in zip(real_feats, fake_feats):
                loss += self.l1(ff, rf.detach())
                n    += 1
        return self.weight * loss / max(n, 1)


# ----------------------------------------------------------
# Charbonnier Loss
# ----------------------------------------------------------

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


# ----------------------------------------------------------
# L1 Loss
# ----------------------------------------------------------

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)


# ----------------------------------------------------------
# SSIM Loss
# ----------------------------------------------------------

class SSIMLoss(nn.Module):
    """
    Standard single-scale SSIM loss with Gaussian window (Wang et al. 2004).

    Does NOT use pytorch_msssim — that library's internal pow(tensor, tensor)
    (PowBackward1) produces NaN gradients when cs_map <= 0 for a fresh model.

    Uses a fixed 11x11 Gaussian kernel (sigma=1.5), matching the original paper.
    Inputs expected in [-1, 1] (network Tanh output).
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        # Build 1-D Gaussian, then outer-product to 2-D, replicate for 3 channels
        coords  = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g1d     = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g1d     = g1d / g1d.sum()
        g2d     = g1d.unsqueeze(1) * g1d.unsqueeze(0)          # (ws, ws)
        kernel  = g2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # (3,1,ws,ws)
        self.register_buffer("kernel", kernel)
        self.pad = window_size // 2

    def _gauss(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-channel Gaussian filter."""
        return F.conv2d(x, self.kernel, padding=self.pad, groups=3)

    def forward(self, pred, target):
        p = (pred.float()   + 1.0) * 0.5   # [-1,1] -> [0,1]
        t = (target.float() + 1.0) * 0.5

        mu_p  = self._gauss(p)
        mu_t  = self._gauss(t)
        mu_p2 = mu_p * mu_p
        mu_t2 = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sig_p  = self._gauss(p.pow(2)) - mu_p2
        sig_t  = self._gauss(t.pow(2)) - mu_t2
        sig_pt = self._gauss(p * t) - mu_pt

        # fp32 subtraction can give tiny negatives — clamp variance to 0
        sig_p = sig_p.clamp(min=0.0)
        sig_t = sig_t.clamp(min=0.0)

        num = (2.0 * mu_pt + self.C1) * (2.0 * sig_pt + self.C2)
        den = (mu_p2 + mu_t2 + self.C1) * (sig_p + sig_t + self.C2)
        # den > 0 always: C1 and C2 are positive constants

        return 1.0 - (num / den).mean()


# ----------------------------------------------------------
# VGG Perceptual Loss
# ----------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    VGG16 perceptual loss on relu1_2, relu2_2, relu3_3 feature maps.

    Why it pushes SSIM higher than pixel losses:
      VGG features encode structural patterns (edges, textures, shapes)
      at multiple scales.  Matching these intermediate representations
      forces the network to produce structurally faithful outputs, which
      directly correlates with SSIM improvement.  Pixel-level losses
      (L1, SSIM) average over spatial locations and cannot capture this.

    Input: [-1, 1] RGB tensors (network Tanh output).
    Features extracted in float32 — VGG is not AMP-safe at fp16.
    VGG weights are frozen; no gradient flows into them.

    layer_weights: relative weight for each feature level.
      Finer layers (relu1_2) supervise texture/detail,
      coarser layers (relu3_3) supervise structure/semantics.
      Default equal weighting; increase relu3_3 weight to
      prioritise global structural fidelity.
    """

    def __init__(self, layer_weights=(1.0, 1.0, 1.0)):
        super().__init__()
        import torchvision.models as models
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except AttributeError:                          # torchvision < 0.13
            vgg = models.vgg16(pretrained=True)

        feats = vgg.features
        self.slice1 = nn.Sequential(*feats[:4])         # relu1_2
        self.slice2 = nn.Sequential(*feats[4:9])        # relu2_2
        self.slice3 = nn.Sequential(*feats[9:16])       # relu3_3

        for p in self.parameters():
            p.requires_grad = False

        self.layer_weights = layer_weights
        self.w_sum         = sum(layer_weights)

        # ImageNet normalization (applied after [-1,1] -> [0,1])
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x):
        x = (x.float() + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        p = self._preprocess(pred)
        t = self._preprocess(target)

        loss = 0.0
        p1 = self.slice1(p);  t1 = self.slice1(t)
        loss += self.layer_weights[0] * F.l1_loss(p1, t1.detach())

        p2 = self.slice2(p1); t2 = self.slice2(t1)
        loss += self.layer_weights[1] * F.l1_loss(p2, t2.detach())

        p3 = self.slice3(p2); t3 = self.slice3(t2)
        loss += self.layer_weights[2] * F.l1_loss(p3, t3.detach())

        return loss / self.w_sum


# ----------------------------------------------------------
# Combined Loss  (SR + Colorization upgrade)
# ----------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Loss stack tuned for 256x256 IR -> 512x512 RGB SR colorization.

    Components
    ----------
    SmoothL1 + MultiScaleEdgeCharbonnier  -- always active (stable from epoch 1)
    SSIMLoss                              -- always active
    FFTFrequencyLoss                      -- ramped in after warmup_epochs
    LabColorLoss                          -- ramped in after warmup_epochs

    For plain Attention U-Net, set warmup_epochs=20 (default) so the network
    first learns basic structure via L1+Edge+SSIM, then FFT and Lab color
    losses activate gradually. For GAN training the discriminator stabilises
    early training, so warmup_epochs=0 is fine.

    Call criterion.set_epoch(epoch) once per epoch in the training loop.
    """

    def __init__(
        self,
        l1_weight         = 1.0,
        ssim_weight       = 0.15,
        edge_weight       = 2.0,
        grad_weight       = 0.4,
        freq_weight       = 0.1,
        color_weight      = 0.5,
        perceptual_weight = 0.0,   # 0 = disabled; try 0.1 as first experiment
        warmup_epochs     = 0,     # epochs before Color loss reaches full weight (0 = always active)
    ):
        super().__init__()
        self.grad_weight        = grad_weight
        self.freq_weight        = freq_weight
        self.color_weight       = color_weight
        self.l1_weight          = l1_weight
        self.ssim_weight        = ssim_weight
        self.perceptual_weight  = perceptual_weight
        self.warmup_epochs      = warmup_epochs
        self._epoch             = 0            # updated via set_epoch()

        self.edge_loss  = MultiScaleEdgeCharbonnierLoss(scales=3, edge_weight=edge_weight)
        self.freq_loss  = FFTFrequencyLoss()
        self.color_loss = LabColorLoss()
        self.ssim_loss  = SSIMLoss()
        self.l1_loss    = nn.SmoothL1Loss(beta=1.0, reduction='mean')

        # Perceptual loss — only instantiate VGG when needed (saves ~500MB VRAM)
        if perceptual_weight > 0.0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

    def set_epoch(self, epoch: int):
        """Call once per epoch so FFT/Color weights ramp up after warmup."""
        self._epoch = epoch

    def _ramp(self):
        """Linear ramp from 0 -> 1 over warmup_epochs. Always 1 if warmup=0."""
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, self._epoch / self.warmup_epochs)

    def forward(self, output, target):
        ramp  = self._ramp()
        edge  = self.edge_loss(output, target)  
        l1    = self.l1_loss(output, target)

        w_freq  = ramp * self.freq_weight
        w_color = ramp * self.color_weight
        color = torch.zeros(1, device=target.device)
        if w_color > 0:
            color = self.color_loss(output, target)
        freq = torch.zeros(1, device=target.device)
        if w_freq > 0:
            freq  = self.freq_loss(output, target)
            
        
        ssim = 1.0 - ms_ssim(
            (output.float() + 1.0) * 0.5,
            (target.float() + 1.0) * 0.5,
            data_range=1.0,
            size_average=True,
            #nonnegative_ssim=True   #  prevents cs_map ** weights NaN
        )
        #ssim  = self.ssim_loss(output, target)

        total = (
            self.grad_weight * edge  +
            self.ssim_weight * ssim  +
            self.l1_weight   * l1    +
            w_freq           * freq  +
            w_color          * color
        )

        perc_val = 0.0
        if self.perceptual_loss is not None:
            # Force fp32 — VGG is not AMP-safe
            with torch.cuda.amp.autocast(enabled=False):
                perc = self.perceptual_loss(output.float(), target.float())
            total    = total + self.perceptual_weight * perc
            perc_val = perc.item()

        losses = {
            'EdgeAC': edge.item(),
            'Freq':   freq.item(),
            'Color':  color.item(),
            'SSIM':   ssim.item(),
            'L1':     l1.item(),
            'Percep': perc_val,
            'Ramp':   ramp,
            'PSNR':   calculate_psnr(output, target),
            'Total':  total.item(),
        }
        # print(f"  l1={l1.item():.3f} edge={edge.item():.3f} ssim={ssim.item():.3f} color={color.item():.3f}")
        return total, losses