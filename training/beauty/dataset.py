"""
SCUT-FBP5500 dataset loader.

Reads the official train.txt / test.txt split files, each line of which
looks like:

    Images/AF1.jpg 3.2

(an image path relative to the dataset root, then the mean beauty
score in [1, 5]).

Returns torch tensors plus the float target. Augmentation lives in
`train.py` — this file just handles I/O.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# The official splits live under this subdirectory of the upstream
# repo. Path is awkward (it has a space in it) but we don't want to
# move it.
SPLIT_DIR_RELATIVE = "train_test_files/split_of_60%training and 40%testing"


class SCUTFBP5500(Dataset):
    """Image + beauty-score dataset for SCUT-FBP5500.

    Parameters
    ----------
    root : str | Path
        Path to the cloned SCUT-FBP5500-Database-Release repo.
    split : "train" | "test"
        Which split file to read.
    transform : callable | None
        Albumentations / torchvision transform applied to the PIL image.
        Receives a uint8 numpy array (H, W, 3) RGB and must return a
        torch.Tensor (C, H, W).
    """

    def __init__(self, root, split: str = "train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        split_file = self.root / SPLIT_DIR_RELATIVE / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found at {split_file}. "
                "Make sure you cloned the SCUT-FBP5500-Database-Release repo "
                "into the path passed via --data-root."
            )

        # Parse "rel/path/to/img.jpg score" pairs.
        self.samples = []
        with split_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # Path can contain spaces in theory; score is always last.
                rel_path = " ".join(parts[:-1])
                score = float(parts[-1])
                self.samples.append((rel_path, score))

        if not self.samples:
            raise RuntimeError(f"No samples loaded from {split_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, score = self.samples[idx]
        img_path = self.root / rel_path
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            # Albumentations transforms expect a dict.
            transformed = self.transform(image=img)
            img = transformed["image"]
        else:
            # Bare fallback: HWC uint8 → CHW float tensor in [0, 1].
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, torch.tensor(score, dtype=torch.float32)
