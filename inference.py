"""
inference.py
--------
Inference: single IR image → RGB SR colorization (256×256 → 512×512)

Usage
-----
    python infer.py --checkpoint runs/exp_attunet_gs_2/best.pth \
                    --input      path/to/ir.png \
                    --output     result.png

    # CPU-only
    python infer.py --checkpoint best.pth --input ir.png --no-cuda

    # Different base_channels (default 64)
    python infer.py --checkpoint best.pth --input ir.png --base-channels 64
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import os
from PIL import Image

from models.model_factory import build_model


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, base_channels: int = 64,
               device: str = "cuda") -> torch.nn.Module:
    cfg = {
        "model": {
            "type":           "attention_unet",
            "in_channels":    1,
            "out_channels":   3,
            "base_channels":  base_channels,
            "grad_checkpoint": False,
            "multi_kernel":   False,
        }
    }
    model = build_model(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle various checkpoint formats
    state = (
        ckpt.get("model_state_dict") or
        ckpt.get("state_dict")       or
        ckpt
    )
    model.load_state_dict(state)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    psnr  = ckpt.get("best_psnr", ckpt.get("psnr", "?"))
    print(f"Loaded checkpoint  epoch={epoch}  psnr={psnr}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Pre / post-processing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> torch.Tensor:
    """
    Load IR image → (1, 1, 256, 256) float32 tensor in [-1, 1].
    Accepts any single-channel or RGB image (converts to grayscale).
    """
    img = Image.open(image_path).convert("L")           # grayscale
    img = img.resize((256, 256), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0 # [0,255] → [-1,1]
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,256,256)


def postprocess(output: torch.Tensor) -> Image.Image:
    """
    Convert model output (1, 3, 512, 512) in [-1, 1] → PIL RGB image.
    """
    arr = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DrishtiNet inference")
    parser.add_argument("--checkpoint",    required=True,  help="Path to .pth checkpoint")
    parser.add_argument("--input",         required=True,  help="Input IR image")
    parser.add_argument("--output",        default="output.png", help="Output RGB image")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--no-cuda",       action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, args.base_channels, device)

    # Preprocess
    tensor = preprocess(args.input).to(device)
    print(f"Input : {args.input}  to  tensor {tuple(tensor.shape)}")

    # Inference
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            output = model(tensor)

    # Postprocess & save
    result = postprocess(output)
    out_path = Path(os.path.join(args.output, os.path.basename(args.checkpoint[:-4]) + "_" + os.path.basename(args.input)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"Output: {out_path}  ({result.size[0]}×{result.size[1]} RGB)")


if __name__ == "__main__":
    main()