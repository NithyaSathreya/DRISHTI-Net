"""
dataset.py — IR -> RGB SR dataset (256x256 IR, 512x512 RGB)
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class NIRRGBDataset(Dataset):

    def __init__(self, root, split="train", image_size=256, rgb_size=None,
                 normalize=True, augment=False):
        self.root       = root
        self.split      = split
        self.ir_size    = image_size
        self.rgb_size   = rgb_size if rgb_size is not None else image_size
        self.normalize  = normalize
        self.augment    = augment

        self.nir_dir = os.path.join(root, split, "IR")
        self.rgb_dir = os.path.join(root, split, "RGB")

        if not os.path.isdir(self.nir_dir):
            raise RuntimeError(f"{self.nir_dir} not found.")
        if not os.path.isdir(self.rgb_dir):
            raise RuntimeError(f"{self.rgb_dir} not found.")

        self.files = sorted([
            f for f in os.listdir(self.nir_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ])
        if len(self.files) == 0:
            raise RuntimeError("No images found.")

        print(f"{split}: {len(self.files)} pairs  IR={self.ir_size}  RGB={self.rgb_size}")

    def __len__(self):
        return len(self.files)

    def _augment(self, nir, rgb):
        if random.random() < 0.5:
            nir = TF.hflip(nir)
            rgb = TF.hflip(rgb)
        if random.random() < 0.5:
            nir = TF.vflip(nir)
            rgb = TF.vflip(rgb)
        angle = random.choice([0, 90, 180, 270])
        nir = TF.rotate(nir, angle)
        rgb = TF.rotate(rgb, angle)
        return nir, rgb

    def __getitem__(self, index):
        filename = self.files[index]
        nir = Image.open(os.path.join(self.nir_dir, filename)).convert("L")
        rgb = Image.open(os.path.join(self.rgb_dir, filename)).convert("RGB")

        # Resize independently: IR -> ir_size, RGB -> rgb_size
        nir = TF.resize(nir, (self.ir_size,  self.ir_size))
        rgb = TF.resize(rgb, (self.rgb_size, self.rgb_size))

        if self.augment:
            nir, rgb = self._augment(nir, rgb)

        nir = TF.to_tensor(nir)
        rgb = TF.to_tensor(rgb)

        if self.normalize:
            nir = TF.normalize(nir, mean=[0.5],           std=[0.5])
            rgb = TF.normalize(rgb, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #if self.normalize:
        #    nir_mean = nir.mean()
        #    nir_std  = nir.std().clamp(min=0.1)
        #    nir = (nir - nir_mean) / nir_std
        #    nir = nir.clamp(-3.0, 3.0)
        #    rgb = TF.normalize(rgb, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return {"nir": nir, "rgb": rgb, "filename": filename}