"""
common.py
---------

Common utility functions used throughout MSAR-Net.

Author : Nithya
"""

import os
import random
import time
import numpy as np
import torch
import yaml


# ---------------------------------------------------------
# Random Seed
# ---------------------------------------------------------

def seed_everything(seed=42):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Create Directory
# ---------------------------------------------------------

def make_dir(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# Current Learning Rate
# ---------------------------------------------------------

def get_lr(optimizer):
    """
    Return current learning rate.
    """
    return optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------
# Count Trainable Parameters
# ---------------------------------------------------------

def count_parameters(model):
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------
# Model Size (Millions)
# ---------------------------------------------------------

def count_parameters_m(model):
    """
    Return trainable parameters in millions.
    """
    return count_parameters(model) / 1e6


# ---------------------------------------------------------
# Format Time
# ---------------------------------------------------------

def format_time(seconds):
    """
    Convert seconds to HH:MM:SS.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ---------------------------------------------------------
# Timestamp
# ---------------------------------------------------------

def timestamp():
    """
    Current timestamp string.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")



def save_args(args, filename):
    """
    Save argparse arguments to a YAML file.
    """

    with open(filename, "w") as f:
        yaml.safe_dump(
            vars(args),
            f,
            default_flow_style=False,
            sort_keys=False
        )