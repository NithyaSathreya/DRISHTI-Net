"""
visualization.py — Visualization utilities (handles IR/RGB size mismatch)
"""

import csv
import random
from itertools import zip_longest

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from utils.image_utils import denormalize


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss(train_vals, val_vals, save_path, label):
    plt.figure(figsize=(8, 5))
    plt.plot(train_vals, label="Train", linewidth=2)
    plt.plot(val_vals,   label="Val",   linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel(label); plt.title(label)
    plt.grid(True); plt.legend()
    save_figure(save_path)


def plot_metric(vals, save_path, label, color="tab:blue"):
    """Single-series metric plot (e.g. val SSIM, val LPIPS).
    Skips entries that are None, empty string, or NaN (epochs not computed)."""
    import math

    def _valid(v):
        if v is None or v == "":
            return False
        try:
            return not math.isnan(float(v))
        except (TypeError, ValueError):
            return False

    epochs = [i  for i, v in enumerate(vals) if _valid(v)]
    clean  = [float(v) for v in vals              if _valid(v)]
    if not clean:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, clean, color=color, linewidth=2, marker="o", markersize=3)
    plt.xlabel("Epoch"); plt.ylabel(label); plt.title(label)
    plt.grid(True)
    save_figure(save_path)


def plot_lr(learning_rate, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(learning_rate, linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("LR"); plt.title("Learning Rate")
    plt.grid(True)
    save_figure(save_path)


def _prep_nir(nir_tensor, target_size=None):
    """Denormalize, expand to 3ch, optionally upsample to target_size (H, W)."""
    nir = denormalize(nir_tensor)
    if nir.shape[0] == 1:
        nir = nir.repeat(3, 1, 1)
    if target_size is not None and nir.shape[-2:] != torch.Size(target_size):
        nir = F.interpolate(nir.unsqueeze(0), size=target_size,
                            mode="bilinear", align_corners=True).squeeze(0)
    return nir


@torch.no_grad()
def save_random_test_prediction(model, dataset, device, save_path):
    model.eval()
    idx    = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    nir    = sample["nir"].unsqueeze(0).to(device)
    gt     = sample["rgb"]
    pred   = model(nir).cpu().squeeze(0)

    rgb_hw = gt.shape[-2:]
    nir_d  = _prep_nir(sample["nir"], target_size=rgb_hw)
    gt_d   = denormalize(gt)
    pred_d = denormalize(pred)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(nir_d.permute(1, 2, 0)); ax[0].set_title("Input IR")
    ax[1].imshow(gt_d.permute(1, 2, 0));  ax[1].set_title("Ground Truth RGB")
    ax[2].imshow(pred_d.permute(1, 2, 0));ax[2].set_title("Predicted RGB")
    for a in ax: a.axis("off")
    save_figure(save_path)
    print(f"Saved: {save_path}")


@torch.no_grad()
def plot_random_predictions(model, dataset, device, save_path, n=4):
    model.eval()
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, ax = plt.subplots(n, 3, figsize=(10, 4 * n))
    if n == 1:
        ax = [ax]

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        nir    = sample["nir"].unsqueeze(0).to(device)
        gt     = sample["rgb"]
        pred   = model(nir).cpu().squeeze(0)

        rgb_hw = gt.shape[-2:]
        nir_d  = _prep_nir(sample["nir"], target_size=rgb_hw)
        gt_d   = denormalize(gt)
        pred_d = denormalize(pred)

        ax[row][0].imshow(nir_d.permute(1, 2, 0));  ax[row][0].set_title("IR")
        ax[row][1].imshow(gt_d.permute(1, 2, 0));   ax[row][1].set_title("GT")
        ax[row][2].imshow(pred_d.permute(1, 2, 0)); ax[row][2].set_title("Pred")
        for c in range(3): ax[row][c].axis("off")

    save_figure(save_path)


def save_history(history, filename):
    rows = zip_longest(*history.values(), fillvalue="")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(rows)


def load_history(filename):
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        return header, list(reader)