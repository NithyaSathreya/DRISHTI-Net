"""
train.py -- IR -> RGB SR Colorization training
Supports: attention_unet | gan_unet, loss v1/v2/v3
Memory tips:
  --amp              : mixed precision (fp16), ~halves VRAM
  --batch_size 2 --grad_accum 2  : same effective batch of 4, half VRAM per step
"""

import os
import argparse
import time
import warnings
import torch
import yaml
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from datasets.dataset import NIRRGBDataset
from losses.losses import CombinedLoss
from models.model_factory import build_model, build_optimizers, is_gan
from utils.common import seed_everything, get_lr, count_parameters_m, save_args
from utils.metrics import calculate_ssim, calculate_lpips
from utils.visualization import plot_loss, plot_metric, plot_lr, save_history, save_random_test_prediction


def parse_args():
    # --config pre-pass: load yaml defaults before argparse sees the rest
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    known, _ = pre.parse_known_args()
    yaml_defaults = {}
    if known.config:
        with open(known.config) as f:
            yaml_defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str,   default="",
                        help="Path to yaml config file. CLI args override yaml values.")
    parser.add_argument("--dataset",       type=str,   required=False,  default="")
    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--ir_size",       type=int,   default=256)
    parser.add_argument("--rgb_size",      type=int,   default=512)
    parser.add_argument("--lr",            type=float, default=5e-4,
                        help="Peak LR. Recommended: 5e-4 for attention_unet, 2e-4 for gan_unet.")
    parser.add_argument("--weight_decay",  type=float, default=1e-2,
                        help="Weight decay applied to conv/linear weights only "
                             "(norm params and biases are always excluded). "
                             "Safe range: 1e-3 to 1e-1 for both Adam and AdamW.")
    parser.add_argument("--optimizer",     type=str,   default="adam",
                        choices=["adam", "adamw"],
                        help="Optimizer: 'adam' (L2 reg, coupled) or 'adamw' "
                             "(decoupled weight decay). Both use param groups "
                             "that exclude norm/bias from weight decay. "
                             "AdamW recommended if Adam still plateaus at epoch 10.")
    parser.add_argument("--lr_warmup",     type=int,   default=5,
                        help="Linear LR warmup epochs (ramp from lr/10 to lr, then cosine).")
    parser.add_argument("--workers",       type=int,   default=4)
    parser.add_argument("--base_channels", type=int,   default=64)
    parser.add_argument("--model_type",    type=str,   default="attention_unet",
                        choices=["attention_unet", "gan_unet"])
    parser.add_argument("--adv_weight",    type=float, default=0.1,
                        help="GAN adversarial loss weight (gan_unet only). "
                             "0.1 gives G strong enough adversarial signal without "
                             "destabilizing reconstruction. Default: 0.1.")
    parser.add_argument("--fm_weight",     type=float, default=10.0,
                        help="Feature matching loss weight (gan_unet only). "
                             "Pix2PixHD standard is 10.0. Provides stable gradient "
                             "signal to G even when D is imperfect. Default: 10.0.")
    parser.add_argument("--checkpoint_dir",type=str,   default="./checkpoints")
    parser.add_argument("--resume",        type=str,   default="")
    parser.add_argument("--save_freq",     type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--exp_run_name",  type=str,   default="exp_0")
    parser.add_argument("--gpu",           type=str,   default="0",
                        help="GPU device(s), e.g. '0' or '0,1'. "
                             "Multiple GPUs use DataParallel (no DDP).")
    parser.add_argument("--amp",            action="store_true",
                        help="Enable automatic mixed precision (fp16). Reduces VRAM ~50%%.")
    parser.add_argument("--grad_checkpoint", action="store_true",
                        help="Gradient checkpointing on encoder/decoder. "
                             "Trades ~30%% compute for ~40%% less activation VRAM.")
    parser.add_argument("--grad_accum",    type=int,   default=1,
                        help="Gradient accumulation steps. "
                             "Effective batch = batch_size * grad_accum.")
    parser.add_argument("--lpips_freq",    type=int,   default=5,
                        help="Compute LPIPS on validation every N epochs (slow). 0 = never.")
    parser.add_argument("--warmup_epochs", type=int,   default=0,
                        help="Epochs before LabColor loss reaches full weight (v3 only). "
                             "Default 0 (no warmup). FFT loss is disabled regardless.")
    parser.add_argument("--d_lr",          type=float, default=5e-5,
                        help="Discriminator LR (gan_unet only). Should be ~4x lower than G LR "
                             "to prevent D from outrunning G. Default: 5e-5.")
    parser.add_argument("--gan_warmup_epochs", type=int, default=5,
                        help="Epochs to train G with reconstruction loss only before "
                             "enabling D and adversarial losses. Default: 5.")
    parser.add_argument("--cosine_T0",       type=int,   default=40,
                        help="CosineAnnealingWarmRestarts period in epochs. "
                             "LR resets to peak every T0 epochs. "
                             "Default 40 gives 3 cycles over 120 epochs.")
    parser.add_argument("--multi_kernel",    action="store_true",
                        help="Use multi-kernel fusion (1x1+3x3+5x5) in ConvBNReLU blocks "
                             "throughout encoder, decoder, and SR head.")
    parser.add_argument("--grad_clip",          type=float, default=1.0,
                        help="Max gradient norm for clip_grad_norm_. "
                             "Lower (e.g. 0.5) if training collapses at peak LR. "
                             "0 disables clipping entirely.")
    parser.add_argument("--perceptual_weight",  type=float, default=0.0,
                        help="Weight for VGG16 perceptual loss (relu1_2, relu2_2, relu3_3). "
                             "0 = disabled (default). Start with 0.1; "
                             "increase to 0.3 if SSIM plateaus. VGG runs fp32 inside AMP.")
    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args()
    if not args.dataset:
        parser.error("--dataset is required (either via --config yaml or CLI)")
    return args


def build_dataloader(args):
    train_ds = NIRRGBDataset(args.dataset, "train", args.ir_size, args.rgb_size,
                             normalize=True, augment=True)
    val_ds   = NIRRGBDataset(args.dataset, "test",  args.ir_size, args.rgb_size,
                             normalize=True, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader, train_ds, val_ds


def _build_model(args, device):
    cfg = {
        "model": {
            "type":            args.model_type,
            "in_channels":     1,
            "out_channels":    3,
            "base_channels":   args.base_channels,
            "grad_checkpoint": args.grad_checkpoint,
            "multi_kernel":    args.multi_kernel,
            "gan": {
                "d_base_channels": args.base_channels,
                "adv_weight":      args.adv_weight,
                "fm_weight":       args.fm_weight,
                "lr":              args.lr,
                "betas":           [0.5, 0.999],
            }
        },
        "training": {"lr": args.lr, "betas": [0.9, 0.999],
                     "weight_decay": args.weight_decay,
                     "optimizer":    getattr(args, "optimizer", "adam")}
    }
    model = build_model(cfg).to(device)
    optimizers = build_optimizers(model, cfg)
    return model, optimizers, cfg


def build_loss(args):

    return CombinedLoss(
        grad_weight        = 0.2,
        ssim_weight        = 5.0,
        l1_weight          = 0.5,
        freq_weight        = 0.0,    # FFT causes regression — MultiScaleEdgeCharbonnier covers HF
        color_weight       = 0.2,
        perceptual_weight  = getattr(args, 'perceptual_weight', 0.0),
        warmup_epochs      = 0,
    )


def build_scheduler(optimizers, args):
    # CosineAnnealingWarmRestarts: resets LR to peak every T_0 epochs.
    # Prevents the model from settling into a local minimum under a
    # monotonically decaying schedule.  T_0=40 gives 3 full restart
    # cycles over 120 epochs (at epochs 40, 80, 120).
    # eta_min=1e-6 keeps LR above zero at each trough.
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizers[0], T_0=args.cosine_T0, T_mult=1, eta_min=1e-6)
    if args.lr_warmup > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizers[0],
            start_factor=0.1, end_factor=1.0,
            total_iters=args.lr_warmup)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizers[0], schedulers=[warmup, cosine],
            milestones=[args.lr_warmup])
    return cosine


# ----------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------

def save_checkpoint(model, optimizers, scheduler, scaler, epoch, best_loss, filename, gan):
    def _sd(m):
        return (m.module if isinstance(m, torch.nn.DataParallel) else m).state_dict()

    state = {
        "epoch":              epoch,
        "best_loss":          best_loss,
        "scheduler_state_dict": scheduler.state_dict(),
        "optimizer_state_dict": optimizers[0].state_dict(),
        "scaler_state_dict":  scaler.state_dict(),
    }

    if gan:
        state["model_state_dict"]         = _sd(model.generator)
        state["discriminator_state_dict"] = _sd(model.discriminator)
        state["optimizer_D_state_dict"]   = optimizers[1].state_dict()
    else:
        state["model_state_dict"] = _sd(model)

    torch.save(state, filename)


def load_checkpoint(model, optimizers, scheduler, scaler, device, filename, gan):
    ckpt = torch.load(filename, map_location=device)

    # ── 1. Extract state dict ─────────────────────────────────────────────────
    first_key = next(iter(ckpt))
    if first_key in ('epoch', 'best_loss', 'model_state_dict',
                     'generator', 'generator_state_dict'):
        state = ckpt.get('model_state_dict') or ckpt.get('generator_state_dict')
    else:
        state = ckpt   # raw state_dict saved directly
        ckpt  = {}

    # ── 2. Auto-remap 'module.' prefix ──────────────────────────────────────
    #    Compare against the model's own keys and add/strip as needed.
    model_keys   = set(next(iter([model.state_dict()])).keys())
    state_keys   = set(state.keys())
    model_has_dp = any(k.startswith('module.') for k in model_keys)
    state_has_dp = any(k.startswith('module.') for k in state_keys)

    if model_has_dp and not state_has_dp:
        state = {'module.' + k: v for k, v in state.items()}   # add prefix
    elif not model_has_dp and state_has_dp:
        state = {k[len('module.'):]: v for k, v in state.items()}  # strip prefix

    # ── 3. Load ───────────────────────────────────────────────────────────────
    is_gan_ckpt = 'discriminator_state_dict' in ckpt

    if gan and not is_gan_ckpt:
        # Warm-start: load attention_unet weights → GAN generator
        # Generator may or may not be DataParallel-wrapped itself
        gen = model.generator
        gen_keys = set(gen.state_dict().keys())
        gen_has_dp = any(k.startswith('module.') for k in gen_keys)
        # state was already remapped for full model; strip 'module.' for generator
        gen_state = state
        if model_has_dp:
            # keys are 'module.generator.xxx' — keep only generator sub-keys
            gen_state = {k[len('module.generator.'):]: v
                         for k, v in state.items()
                         if k.startswith('module.generator.')}
        if gen_has_dp and not any(k.startswith('module.') for k in gen_state):
            gen_state = {'module.' + k: v for k, v in gen_state.items()}
        gen.load_state_dict(gen_state)
        print(f"[Resume] attention_unet → GAN generator  epoch={ckpt.get('epoch','?')}")

    elif gan and is_gan_ckpt:
        model.load_state_dict(state)
        disc_state = ckpt['discriminator_state_dict']
        model.discriminator.load_state_dict(disc_state)
        if optimizers and 'optimizer_state_dict' in ckpt:
            optimizers[0].load_state_dict(ckpt['optimizer_state_dict'])
        if len(optimizers) > 1 and 'optimizer_D_state_dict' in ckpt:
            optimizers[1].load_state_dict(ckpt['optimizer_D_state_dict'])

    else:
        model.load_state_dict(state)
        if optimizers and 'optimizer_state_dict' in ckpt:
            optimizers[0].load_state_dict(ckpt['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    return ckpt.get('epoch', 0) + 1, ckpt.get('best_loss', float('inf'))


def save_best_weights(model, path, gan):
    if gan:
        g = model.generator
        sd = (g.module if isinstance(g, torch.nn.DataParallel) else g).state_dict()
    else:
        sd = (model.module if isinstance(model, torch.nn.DataParallel) else model).state_dict()
    torch.save(sd, path)


# ----------------------------------------------------------
# Train one epoch  (AMP + gradient accumulation)
# ----------------------------------------------------------

def _clip(params, max_norm):
    """clip_grad_norm_ wrapper — skipped when max_norm=0 (disabled)."""
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def train_one_epoch(model, loader, criterion, optimizers, device, epoch, gan,
                    scaler, use_amp, grad_accum, gan_warmup_epochs=5, grad_clip=1.0):
    model.train()
    totals = {}
    n = len(loader)
    start = time.time()

    # Zero grads at epoch start
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    for batch_idx, sample in enumerate(loader):
        nir = sample["nir"].to(device, non_blocking=True)
        rgb = sample["rgb"].to(device, non_blocking=True)
        is_accum_step = ((batch_idx + 1) % grad_accum != 0) and \
                        ((batch_idx + 1) != n)

        if gan:
            opt_G, opt_D = optimizers
            in_warmup = epoch <= gan_warmup_epochs

            if in_warmup:
                # Warmup: G trains on reconstruction loss only — no D, no adversarial.
                # Gives G a head start so D doesn't dominate from epoch 1.
                with autocast('cuda', enabled=use_amp):
                    fake = model.generator(nir)
                g_loss, losses = criterion(fake, rgb)
                g_loss = g_loss / grad_accum

                total_val = losses.get("Total", losses.get("loss", 0.0))
                if not torch.isfinite(torch.tensor(float(total_val))):
                    print(f"WARNING: non-finite loss={float(total_val):.4f} at "
                          f"epoch {epoch} step {batch_idx+1} (GAN warmup) — skipping batch.")
                    opt_G.zero_grad(set_to_none=True)
                    losses["D_loss"] = 0.0
                    continue

                scaler.scale(g_loss).backward()

                if not is_accum_step:
                    scaler.unscale_(opt_G)
                    _clip(model.generator.parameters(), grad_clip)
                    scaler.step(opt_G)
                    scaler.update()
                    opt_G.zero_grad(set_to_none=True)

                losses["D_loss"] = 0.0

            else:
                # Full GAN — D step then G step.
                # Balance via LR asymmetry (d_lr=5e-5 vs g_lr=2e-4).
                # Spectral norm + label smoothing (real=0.9) prevent D from dominating.
                # Compute both losses first, guard both before any backward.
                # This ensures neither backward runs if either loss is bad.
                with autocast('cuda', enabled=use_amp):
                    d_loss = model.discriminator_step(nir, rgb)

                with autocast('cuda', enabled=use_amp):
                    g_loss, losses = model.generator_step(nir, rgb, criterion)
                    g_loss = g_loss / grad_accum

                total_val = losses.get("Total", losses.get("loss", 0.0))
                if not torch.isfinite(d_loss) or \
                   not torch.isfinite(torch.tensor(float(total_val))):
                    print(f"WARNING: non-finite loss (d={d_loss.item():.4f} "
                          f"g={float(total_val):.4f}) at epoch {epoch} "
                          f"step {batch_idx+1} — skipping batch.")
                    opt_G.zero_grad(set_to_none=True)
                    opt_D.zero_grad(set_to_none=True)
                    continue

                scaler.scale(d_loss / grad_accum).backward()
                scaler.scale(g_loss).backward()

                if not is_accum_step:
                    scaler.unscale_(opt_D)
                    _clip(model.discriminator.parameters(), grad_clip)
                    scaler.step(opt_D)
                    opt_D.zero_grad(set_to_none=True)

                    scaler.unscale_(opt_G)
                    _clip(model.generator.parameters(), grad_clip)
                    scaler.step(opt_G)
                    scaler.update()
                    opt_G.zero_grad(set_to_none=True)

                losses["D_loss"] = d_loss.item()

        else:
            (opt,) = optimizers
            with autocast('cuda', enabled=use_amp):
                pred = model(nir)
                # ADD THIS:
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print(f"[E{epoch} B{batch_idx}] NaN/Inf in model output!")
            loss, losses = criterion(pred, rgb)
            loss = loss / grad_accum

            # NaN/Inf guard BEFORE backward — prevents corrupted weights.
            # If guard fires after backward, NaN gradients have already been
            # applied (or GradScaler silently skipped the step, halving the
            # scale repeatedly until gradients underflow).
            total_val = losses.get("Total", losses.get("loss", 0.0))
            if not torch.isfinite(torch.tensor(float(total_val))):
                print(f"WARNING: non-finite loss={float(total_val):.4f} at "
                      f"epoch {epoch} step {batch_idx+1} — skipping batch.")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if not is_accum_step:
                scaler.unscale_(opt)
                _clip(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                # print(f"  scale={scaler.get_scale():.1f}")
                opt.zero_grad(set_to_none=True)

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + (v if isinstance(v, float) else v)

        if (batch_idx + 1) % 20 == 0:
            print(f"Epoch [{epoch}] Step [{batch_idx+1}/{n}] "
                  f"Loss {totals.get('Total', 0.0)/(batch_idx+1):.4f}")

    return {k: v / n for k, v in totals.items()}, time.time() - start


# ----------------------------------------------------------
# Validate
# ----------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, compute_lpips=False):
    import math
    model.eval()
    totals = {}
    ssim_sum = lpips_sum = 0.0
    ssim_count = lpips_count = 0
    n = len(loader)
    for sample in loader:
        nir  = sample["nir"].to(device)
        rgb  = sample["rgb"].to(device)
        with autocast('cuda', enabled=use_amp):
            pred = model(nir)
        _, losses = criterion(pred, rgb)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + (v if isinstance(v, float) else v)

        pred_f = pred.float()
        rgb_f  = rgb.float()

        # Skip NaN predictions so one bad sample doesn't corrupt the averages.
        if torch.isnan(pred_f).any():
            continue

        s = calculate_ssim(pred_f, rgb_f)
        if not math.isnan(s):
            ssim_sum   += s
            ssim_count += 1

        if compute_lpips:
            lp = calculate_lpips(pred_f, rgb_f, device=device)
            if not math.isnan(lp):
                lpips_sum   += lp
                lpips_count += 1

    result = {k: v / n for k, v in totals.items()}
    result["SSIM_metric"]  = ssim_sum  / ssim_count  if ssim_count  > 0 else float("nan")
    result["LPIPS_metric"] = (lpips_sum / lpips_count if lpips_count > 0 else float("nan")) \
                              if compute_lpips else None
    return result


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    # Suppress third-party deprecation noise we don't control
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="The epoch parameter in `scheduler.step\\(\\)`")
    # lpips internally loads torchvision models with the old pretrained= API
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="The parameter 'pretrained' is deprecated")
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="Arguments other than a weight enum or")
    args = parse_args()
    seed_everything(args.seed)

    exp_folder    = os.path.join(args.checkpoint_dir, args.exp_run_name)
    result_folder = os.path.join("results", args.exp_run_name)
    os.makedirs(exp_folder,    exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    save_args(args, os.path.join(result_folder, "args.yaml"))
    save_args(args, os.path.join(exp_folder,    "args.yaml"))  # also saved in checkpoint dir

    gpu_ids = [int(g) for g in args.gpu.split(",")]
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
        gpu_ids = []

    use_amp   = args.amp and device.type == "cuda"
    #scaler    = GradScaler(enabled=use_amp)
    scaler = GradScaler(enabled=use_amp, init_scale=64, growth_interval=2000)
    eff_batch = args.batch_size * args.grad_accum

    print(f"Device: {device}  GPUs: {gpu_ids}  Model: {args.model_type}  "
          f"Loss: {args.loss_type}  AMP: {use_amp}  "
          f"EffectiveBatch: {eff_batch} (bs={args.batch_size} x accum={args.grad_accum})")

    train_loader, val_loader, train_data, val_data = build_dataloader(args)
    model, optimizers, cfg = _build_model(args, device)
    # For GAN: rebuild D optimizer at a lower LR so D doesn't outrun G.
    # D_lr = 5e-5 vs G_lr = 2e-4  (4x slower) prevents D from dominating by epoch 2.
    if is_gan(cfg):
        opt_G, _ = optimizers
        opt_D = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=args.d_lr, betas=(0.5, 0.999))
        optimizers = (opt_G, opt_D)
    print(f"Parameters: {count_parameters_m(model):.2f} M")

    # Wrap with DataParallel if multiple GPUs requested
    if len(gpu_ids) > 1:
        if is_gan(cfg):
            model.generator     = torch.nn.DataParallel(model.generator,     device_ids=gpu_ids)
            model.discriminator = torch.nn.DataParallel(model.discriminator, device_ids=gpu_ids)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"DataParallel across GPUs: {gpu_ids}")

    criterion = build_loss(args).to(device)
    scheduler = build_scheduler(optimizers, args)
    gan       = is_gan(cfg)

    start_epoch = 1
    best_loss   = float("inf")
    if args.resume:
        start_epoch, best_loss = load_checkpoint(
            model, optimizers, scheduler, scaler, device, args.resume, gan)

    history = {"train_loss": [], "val_loss": [],
               "train_psnr": [], "val_psnr": [],
               "val_ssim":   [], "val_lpips": [],
               "d_loss":     [],
               "lr":         []}
    # orch.autograd.set_detect_anomaly(True)   # remove after debugging, it's slow
    for epoch in range(start_epoch, args.epochs + 1):
        do_lpips = (args.lpips_freq > 0) and (epoch % args.lpips_freq == 0)

        # Ramp FFT/Color losses in v3 based on current epoch
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)

        train_stats, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizers, device, epoch, gan,
            scaler, use_amp, args.grad_accum,
            gan_warmup_epochs=args.gan_warmup_epochs,
            grad_clip=args.grad_clip)
        val_stats = validate(model, val_loader, criterion, device, use_amp,
                             compute_lpips=do_lpips)

        scheduler.step()
        lr = get_lr(optimizers[0])

        t_loss  = train_stats.get("Total", train_stats.get("loss", 0.0))
        v_loss  = val_stats.get("Total",   val_stats.get("loss",   0.0))
        t_psnr  = train_stats.get("PSNR",  0.0)
        v_psnr  = val_stats.get("PSNR",    0.0)
        v_ssim  = val_stats.get("SSIM_metric", 0.0)
        v_lpips = val_stats.get("LPIPS_metric", None)
        d_loss  = train_stats.get("D_loss", None)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_psnr"].append(t_psnr)
        history["val_psnr"].append(v_psnr)
        history["val_ssim"].append(v_ssim)
        history["val_lpips"].append(v_lpips)
        history["d_loss"].append(d_loss)
        history["lr"].append(lr)

        print(f"Epoch [{epoch}/{args.epochs}] "
              f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  "
              f"PSNR={v_psnr:.2f}  SSIM={v_ssim:.4f}  "
              f"lr={lr:.2e}  time={epoch_time:.1f}s"
              + (f"  d_loss={d_loss:.4f}" if d_loss else ""))

        # Save best weights
        if v_loss < best_loss:
            best_loss = v_loss
            save_best_weights(model, os.path.join(exp_folder, "best_weights.pth"), gan)
            print(f"  -> best model saved (val_loss={best_loss:.4f})")

        # Periodic checkpoint
        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(exp_folder, f"epoch_{epoch:03d}.pth")
            save_checkpoint(model, optimizers, scheduler, scaler,
                            epoch, best_loss, ckpt_path, gan)

        # Plots + history CSV
        save_history(history, os.path.join(result_folder, "history.csv"))
        plot_loss(history["train_loss"], history["val_loss"],
                  os.path.join(result_folder, "loss.png"), "Loss")
        plot_metric(history["val_psnr"], os.path.join(result_folder, "psnr.png"), "PSNR")
        plot_metric(history["val_ssim"], os.path.join(result_folder, "ssim.png"), "SSIM")
        plot_lr(history["lr"],           os.path.join(result_folder, "lr.png"))

        # Validation sample visualisation
        save_random_test_prediction(model, val_data, device,
                                    os.path.join(result_folder, f"pred_epoch_{epoch:03d}.png"))

    print("Training complete.")


if __name__ == "__main__":
    main()