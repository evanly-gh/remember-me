"""
Evaluate a trained beauty regressor against the SCUT-FBP5500 test set.

Prints MSE, MAE, Pearson r, Spearman ρ — the standard metric quartet
used in the SCUT-FBP5500 literature. Use this to sanity-check before
deploying a new checkpoint into face-service.
"""

import argparse

import albumentations as A
import numpy as np
import timm
import torch
from albumentations.pytorch import ToTensorV2
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from dataset import SCUTFBP5500

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--tta",
        action="store_true",
        help="average predictions with horizontal flip (match train.py val TTA)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_tf = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val_ds = SCUTFBP5500(args.data_root, split="test", transform=val_tf)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Build model + load weights. strict=True to catch silent mismatches.
    model = timm.create_model(args.backbone, pretrained=False, num_classes=1)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).squeeze(-1)
            if args.tta:
                preds = (preds + model(torch.flip(imgs, dims=[3])).squeeze(-1)) * 0.5
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    pearson, _ = pearsonr(preds, targets)
    spearman, _ = spearmanr(preds, targets)

    print(f"Test samples: {len(val_ds)}")
    print(f"  MSE:       {mse:.4f}")
    print(f"  MAE:       {mae:.4f}")
    print(f"  Pearson r: {pearson:.4f}")
    print(f"  Spearman ρ: {spearman:.4f}")
    print(
        "\nFor reference, the published academic best on SCUT-FBP5500 "
        "is r=0.8997 with ResNeXt-50 (Liang et al., ICPR 2018)."
    )


if __name__ == "__main__":
    main()
