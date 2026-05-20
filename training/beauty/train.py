"""
Train a ResNet-50 (or any timm backbone) regressor on SCUT-FBP5500.

Single-script training entry point. Usage:

    python train.py \\
        --data-root data/SCUT-FBP5500 \\
        --epochs 25 \\
        --batch-size 64 \\
        --lr 1e-4 \\
        --backbone resnet50 \\
        --out checkpoints/beauty_regressor.pt

What it does:
- Loads SCUT-FBP5500's official 60/40 train/test split via dataset.py.
- Builds a `timm` backbone with a single-output regression head.
- Standard augmentation: horizontal flip, random resized crop,
  color jitter; ImageNet normalisation.
- Adam optimiser + cosine LR schedule; MSE loss.
- Logs train/val MSE + Pearson r every epoch.
- Saves the checkpoint with the best validation Pearson r.

The output .pt is exactly what BeautyAnalyzer expects: a bare
PyTorch state_dict matching the timm model's keys.
"""

import argparse
import os
from pathlib import Path

import albumentations as A
import numpy as np
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SCUTFBP5500

# Standard ImageNet stats. Pre-trained timm backbones expect this norm.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int):
    """Train + val transforms.

    Train side: light augmentation (flip, RRC, color jitter). The
    dataset is small (5,500 images) and heavy augmentation hurts more
    than it helps for regression.

    Val side: deterministic resize + normalise.
    """
    train_tf = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tf, val_tf


def build_model(backbone: str) -> nn.Module:
    """timm backbone with a single regression output."""
    return timm.create_model(backbone, pretrained=True, num_classes=1)


def evaluate(model, loader, device) -> tuple[float, float, float]:
    """Run validation, returning (mse, mae, pearson_r)."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs).squeeze(-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    # Pearson r is the standard SCUT-FBP5500 reporting metric.
    r, _ = pearsonr(preds, targets)
    return mse, mae, float(r)


def main():
    parser = argparse.ArgumentParser(description="SCUT-FBP5500 beauty regressor training")
    parser.add_argument("--data-root", required=True, help="path to cloned SCUT-FBP5500 repo")
    parser.add_argument("--out", default="checkpoints/beauty_regressor.pt")
    parser.add_argument("--backbone", default="resnet50",
                        help="any timm model name (resnet50, convnext_small, …)")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    parser.add_argument(
        "--num-workers", type=int, default=default_workers,
        help="DataLoader workers (defaults to SLURM_CPUS_PER_TASK on Hyak)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = SCUTFBP5500(args.data_root, split="train", transform=train_tf)
    val_ds = SCUTFBP5500(args.data_root, split="test", transform=val_tf)
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model(args.backbone).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_pearson = -np.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = model(imgs).squeeze(-1)
            loss = loss_fn(preds, targets)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            bs = imgs.size(0)
            running += loss.item() * bs
            seen += bs
            pbar.set_postfix(train_mse=running / seen)

        scheduler.step()
        val_mse, val_mae, val_r = evaluate(model, val_loader, device)
        print(
            f"  epoch {epoch:02d}  train_mse={running / max(1, seen):.4f}  "
            f"val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  pearson={val_r:.4f}"
        )

        # Keep the highest-Pearson checkpoint, not lowest-MSE — Pearson
        # is the standard SCUT-FBP5500 metric and is robust to a small
        # constant bias in the predictions.
        if val_r > best_pearson:
            best_pearson = val_r
            torch.save(model.state_dict(), out_path)
            print(f"    ↑ new best pearson={val_r:.4f} → saved to {out_path}")

    print(f"\nBest validation Pearson r: {best_pearson:.4f}")
    print(f"Checkpoint: {out_path}")
    print("Drop it into face-service/models/beauty_regressor.pt and restart "
          "the service to wire it into the pipeline.")


if __name__ == "__main__":
    main()
