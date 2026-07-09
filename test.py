"""
test.py — Inference and evaluation with PSNR / SSIM / LPIPS / FID
FID requires torchmetrics: pip install torchmetrics[image]
LPIPS requires: pip install lpips

Usage
-----
    python test.py --config checkpoints/exp_attunet_v1/args.yaml \
                   --weights checkpoints/exp_attunet_v1/best_weights.pth \
                   --output  results/eval

    # With FID (needs 200+ test images for a meaningful score)
    python test.py --config checkpoints/exp_attunet_v1/args.yaml \
                   --weights checkpoints/exp_attunet_v1/best_weights.pth \
                   --output  results/eval --compute_fid
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from datasets.dataset import NIRRGBDataset
from losses.losses import CombinedLoss_v3
from models.model_factory import build_model, is_gan
from utils.image_utils import create_directory, denormalize, save_prediction, make_comparison
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    _fid_available = True
except ImportError:
    _fid_available = False


# ──────────────────────────────────────────────────────────────────────────────
# Args  (same yaml format as train.py — no separate config module needed)
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DrishtiNet evaluation")
    parser.add_argument("--config",      required=True,  type=str,
                        help="Path to the args.yaml saved during training")
    parser.add_argument("--weights",     required=True,  type=str,
                        help="Path to best_weights.pth or any epoch checkpoint")
    parser.add_argument("--output",      default="results/eval", type=str,
                        help="Directory to write predictions and comparison images")
    parser.add_argument("--split",       default="test", type=str,
                        choices=["train", "test", "val"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--compute_fid", action="store_true",
                        help="Compute FID (recommended with 200+ test images)")
    args = parser.parse_args()

    # Load yaml and merge into args — identical format to train.py
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in cfg.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # Safe defaults for fields that may be absent in older yaml files
    args.ir_size         = getattr(args, "ir_size",         256)
    args.rgb_size        = getattr(args, "rgb_size",         512)
    args.base_channels   = getattr(args, "base_channels",    64)
    args.model_type      = getattr(args, "model_type",       "attention_unet")
    args.grad_checkpoint = getattr(args, "grad_checkpoint",  False)
    args.multi_kernel    = getattr(args, "multi_kernel",     False)
    args.adv_weight      = getattr(args, "adv_weight",       0.1)
    args.fm_weight       = getattr(args, "fm_weight",        10.0)
    args.workers         = getattr(args, "workers",           4)

    if not getattr(args, "dataset", None):
        parser.error("The yaml config must contain a 'dataset' key with the data root path")

    return args


# ──────────────────────────────────────────────────────────────────────────────
# Dataloader
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloader(args):
    dataset = NIRRGBDataset(
        args.dataset, args.split,
        args.ir_size, args.rgb_size,
        normalize=True, augment=False,
    )
    return DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# Handles: raw state dicts, full checkpoints, DataParallel 'module.' prefix
# ──────────────────────────────────────────────────────────────────────────────

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
                "lr":              0.0002,
                "betas":           [0.5, 0.999],
            },
        }
    }
    model = build_model(cfg).to(device)
    gan   = is_gan(cfg)

    raw = torch.load(args.weights, map_location=device)

    # ── Detect checkpoint format ───────────────────────────────────────────────
    # save_best_weights() → raw state dict   (first key is a layer name)
    # save_checkpoint()   → metadata wrapper (first key is 'epoch' / 'model_state_dict' / …)
    first_key = next(iter(raw))
    if first_key in ("epoch", "best_loss", "model_state_dict",
                     "scheduler_state_dict", "optimizer_state_dict"):
        # Full checkpoint: generator weights always stored under "model_state_dict"
        state = raw["model_state_dict"]
        epoch = raw.get("epoch", "?")
    else:
        # Raw state dict saved directly with torch.save(model.state_dict(), path)
        state = raw
        epoch = "?"

    # ── Strip DataParallel 'module.' prefix if present ────────────────────────
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    # ── Load into model ───────────────────────────────────────────────────────
    if gan:
        # state holds AttentionUNet (generator) weights.
        # Verify keys match generator before loading.
        gen_keys   = set(model.generator.state_dict().keys())
        state_keys = set(state.keys())
        if not state_keys.issubset(gen_keys) and not gen_keys.issubset(state_keys):
            # Try stripping a "generator." prefix (edge case from old saves)
            alt = {k[len("generator."):]: v
                   for k, v in state.items() if k.startswith("generator.")}
            if alt:
                state = alt
            else:
                raise RuntimeError(
                    f"Cannot match checkpoint keys to GAN generator.\n"
                    f"  Checkpoint sample: {list(state_keys)[:4]}\n"
                    f"  Generator sample:  {list(gen_keys)[:4]}"
                )
        model.generator.load_state_dict(state, strict=False)
    else:
        # Remap prefix if the live model is DataParallel but checkpoint is not (or vice versa)
        model_keys   = set(model.state_dict().keys())
        model_has_dp = any(k.startswith("module.") for k in model_keys)
        state_has_dp = any(k.startswith("module.") for k in state)
        if model_has_dp and not state_has_dp:
            state = {"module." + k: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

    print(f"Loaded : {args.weights}")
    print(f"  model={args.model_type}  epoch={epoch}  device={device}")
    return model.eval()


# ──────────────────────────────────────────────────────────────────────────────
# FID helper
# ──────────────────────────────────────────────────────────────────────────────

def _to_fid_input(tensor):
    """Convert [-1, 1] tensor → float32 [0, 1] (B, 3, H, W) for FID.

    FrechetInceptionDistance(normalize=True) expects float tensors in [0, 1].
    Passing uint8 here is wrong — Inception would receive values like 200
    instead of 0.78, producing garbage features and a meaningless FID score.
    """
    return denormalize(tensor).clamp(0.0, 1.0).float()


# ──────────────────────────────────────────────────────────────────────────────
# Inference + evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def inference(model, loader, criterion, device, output_dir, compute_fid):
    pred_dir    = os.path.join(output_dir, "prediction")
    compare_dir = os.path.join(output_dir, "comparison")
    create_directory(pred_dir)
    create_directory(compare_dir)

    # FID accumulator
    fid_metric = None
    if compute_fid:
        if not _fid_available:
            print("WARNING: torchmetrics not installed — skipping FID.\n"
                  "  Install with: pip install torchmetrics[image]")
            compute_fid = False
        else:
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    total_loss = total_psnr = total_ssim = total_lpips = 0.0
    count = 0

    for sample in loader:
        nir = sample["nir"].to(device)
        rgb = sample["rgb"].to(device)

        # Filename — dataset may or may not return this key
        if "filename" in sample:
            filename = sample["filename"][0]
        else:
            filename = f"{count + 1:04d}.png"
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filename += ".png"

        # Forward (no autocast — criterion must run in fp32)
        pred = model(nir)

        # Criterion OUTSIDE autocast — avoids the SSIMLoss fp16 NaN bug
        loss, _ = criterion(pred.float(), rgb.float())

        psnr_val  = calculate_psnr(pred,  rgb)
        ssim_val  = calculate_ssim(pred,  rgb)
        lpips_val = calculate_lpips(pred, rgb, device=device)

        if fid_metric is not None:
            fid_metric.update(_to_fid_input(rgb),  real=True)
            fid_metric.update(_to_fid_input(pred), real=False)

        save_prediction(pred, os.path.join(pred_dir,    filename))
        make_comparison(nir,  rgb, pred, os.path.join(compare_dir, filename))

        total_loss  += loss.item()
        total_psnr  += psnr_val
        total_ssim  += ssim_val
        total_lpips += lpips_val
        count += 1

        print(f"[{count:04d}/{len(loader)}] {filename:25s} "
              f"PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}  LPIPS={lpips_val:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Images    : {count}")
    print(f"Avg Loss  : {total_loss  / count:.4f}")
    print(f"Avg PSNR  : {total_psnr  / count:.2f} dB")
    print(f"Avg SSIM  : {total_ssim  / count:.4f}")
    print(f"Avg LPIPS : {total_lpips / count:.4f}  (lower is better)")
    if fid_metric is not None:
        fid_score = fid_metric.compute().item()
        print(f"FID       : {fid_score:.2f}  (lower is better)")
    print("="*60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Weights: {args.weights}  FID: {args.compute_fid}")

    loader    = build_dataloader(args)
    model     = _build_model(args, device)
    criterion = CombinedLoss_v3().to(device)

    inference(model, loader, criterion, device, args.output, args.compute_fid)


if __name__ == "__main__":
    main()