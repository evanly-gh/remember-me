"""
Train a ResNet-50 (or any timm backbone) regressor on SCUT-FBP5500.

Single- or multi-GPU (DDP via torchrun). Usage:

    # Single GPU
    python train.py \\
        --data-root data/SCUT-FBP5500 \\
        --epochs 50 \\
        --batch-size 64 \\
        --lr 1e-4 \\
        --backbone resnet50 \\
        --out checkpoints/beauty_regressor.pt

    # Four GPUs (global batch = batch_size × 4; scale --lr accordingly)
    torchrun --standalone --nproc_per_node=4 train.py \\
        --data-root data/SCUT-FBP5500 \\
        --batch-size 32 \\
        --lr 2e-4 \\
        ...

What it does:
- Loads SCUT-FBP5500's official 60/40 train/test split via dataset.py.
- Builds a `timm` backbone with a single-output regression head.
- Standard augmentation: horizontal flip, random resized crop,
  color jitter; ImageNet normalisation.
- AdamW + linear LR warmup then cosine decay; MSE loss; optional multi-GPU DDP.
- Optional EMA of weights; validation uses EMA weights when enabled.
- Optional horizontal-flip TTA on validation for checkpoint selection.
- Logs train/val MSE + Pearson r every epoch (rank 0).
- Saves the checkpoint with the best validation Pearson r (EMA state_dict
  when EMA is enabled).

The output .pt is exactly what BeautyAnalyzer expects: a bare
PyTorch state_dict matching the timm model's keys.
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from scipy.stats import pearsonr
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import SCUTFBP5500

# Standard ImageNet stats. Pre-trained timm backbones expect this norm.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_main() -> bool:
    return _rank() == 0


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def setup_distributed() -> torch.device:
    """Init NCCL if launched with torchrun. Returns the CUDA device for this process."""
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device(f"cuda:{local_rank}")


def teardown_distributed() -> None:
    if _is_dist():
        dist.destroy_process_group()


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


class ModelEMA:
    """Exponential moving average of model weights (eval / checkpoint selection)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, ema_v in self.shadow.state_dict().items():
            if not ema_v.dtype.is_floating_point:
                ema_v.copy_(msd[k])
            else:
                ema_v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)


def _recommended_ema_decay(*, steps_per_epoch: int) -> float:
    """
    Pick an EMA decay that actually tracks training at the current step rate.

    The old default (0.9999) is fine for very long runs, but with small datasets
    and large global batches (few steps/epoch) it can freeze EMA near init.
    """
    # If you have ~25–50 steps/epoch, 0.99 tracks well; 0.995 is slightly smoother.
    if steps_per_epoch <= 60:
        return 0.99
    if steps_per_epoch <= 200:
        return 0.995
    return 0.999


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tta_hflip: bool,
) -> Tuple[float, float, float]:
    """Run validation, returning (mse, mae, pearson_r)."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs).squeeze(-1)
            if tta_hflip:
                flipped = torch.flip(imgs, dims=[3])
                preds = (preds + model(flipped).squeeze(-1)) * 0.5
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    r, _ = pearsonr(preds, targets)
    return mse, mae, float(r)


def build_lr_scheduler(optimiser, *, epochs: int, warmup_epochs: int):
    """Linear warmup from 0.1×lr → 1×lr, then cosine decay to ~0."""
    warmup_epochs = max(0, min(warmup_epochs, max(0, epochs - 1)))
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimiser, T_max=max(1, epochs))
    cosine_epochs = max(1, epochs - warmup_epochs)
    warmup = LinearLR(
        optimiser, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = CosineAnnealingLR(optimiser, T_max=cosine_epochs)
    return SequentialLR(
        optimiser, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SCUT-FBP5500 beauty regressor training")
    parser.add_argument("--data-root", required=True, help="path to cloned SCUT-FBP5500 repo")
    parser.add_argument("--out", default="checkpoints/beauty_regressor.pt")
    parser.add_argument("--backbone", default="resnet50",
                        help="any timm model name (resnet50, convnext_small, …)")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="linear LR warmup before cosine decay")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="per-GPU batch when using DDP (global batch = this × world size)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    parser.add_argument(
        "--num-workers", type=int, default=default_workers,
        help="DataLoader workers (defaults to SLURM_CPUS_PER_TASK on Hyak)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay; set to 0 to disable EMA")
    parser.add_argument(
        "--no-val-tta",
        action="store_true",
        help="disable horizontal-flip TTA on validation (checkpoint metric)",
    )
    args = parser.parse_args()

    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    device = setup_distributed() if distributed else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if distributed and not torch.cuda.is_available():
        print("ERROR: DDP requested but CUDA is not available.", file=sys.stderr)
        sys.exit(1)

    # Same initial weights on every rank; different seeds mainly affect dropout (none here).
    torch.manual_seed(args.seed + _rank())
    np.random.seed(args.seed + _rank())

    if _is_main():
        print(f"Device: {device}  |  distributed={distributed}  |  world_size={_world_size()}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = SCUTFBP5500(args.data_root, split="train", transform=train_tf)
    val_ds = SCUTFBP5500(args.data_root, split="test", transform=val_tf)
    if _is_main():
        print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.backbone).to(device)
    if distributed:
        model = DDP(model, device_ids=[_local_rank()], output_device=_local_rank())

    core = _unwrap(model)
    steps_per_epoch = max(1, len(train_loader))

    # EMA is very helpful here, but the decay must match the step rate.
    ema_decay = float(args.ema_decay)
    if ema_decay <= 0.0:
        ema_decay = _recommended_ema_decay(steps_per_epoch=steps_per_epoch)
        if _is_main():
            print(f"EMA: enabled (auto) decay={ema_decay} (steps/epoch={steps_per_epoch})")
    else:
        if _is_main():
            print(f"EMA: enabled decay={ema_decay} (steps/epoch={steps_per_epoch})")

    use_ema = ema_decay > 0.0
    ema: Optional[ModelEMA] = ModelEMA(core, decay=ema_decay) if use_ema else None

    optimiser = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(
        optimiser, epochs=args.epochs, warmup_epochs=args.warmup_epochs
    )
    loss_fn = nn.MSELoss()

    out_path = Path(args.out)
    if _is_main():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    best_pearson = -np.inf
    tta = not args.no_val_tta

    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            running = 0.0
            seen = 0
            iterator = train_loader
            if _is_main():
                iterator = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")

            for imgs, targets in iterator:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                preds = model(imgs).squeeze(-1)
                loss = loss_fn(preds, targets)

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                if ema is not None:
                    ema.update(core)

                bs = imgs.size(0)
                running += loss.item() * bs
                seen += bs
                if _is_main() and isinstance(iterator, tqdm):
                    iterator.set_postfix(train_mse=running / max(1, seen))

            scheduler.step()

            val_mse = val_mae = val_r = 0.0
            if _is_main():
                eval_net = ema.shadow if ema is not None else core
                val_mse, val_mae, val_r = evaluate(
                    eval_net, val_loader, device, tta_hflip=tta
                )
                print(
                    f"  epoch {epoch:02d}  train_mse={running / max(1, seen):.4f}  "
                    f"val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  pearson={val_r:.4f}"
                )

                if val_r > best_pearson:
                    best_pearson = val_r
                    to_save = ema.shadow.state_dict() if ema is not None else core.state_dict()
                    torch.save(to_save, out_path)
                    tag = "ema" if ema is not None else "online"
                    print(
                        f"    ↑ new best pearson={val_r:.4f} ({tag}) → saved to {out_path}"
                    )

            if distributed:
                dist.barrier()

        if _is_main():
            print(f"\nBest validation Pearson r: {best_pearson:.4f}")
            print(f"Checkpoint: {out_path}")
            print("Drop it into face-service/models/beauty_regressor.pt and restart "
                  "the service to wire it into the pipeline.")
            if args.img_size != 224:
                print(
                    f"NOTE: trained with --img-size {args.img_size}. Set "
                    f"BEAUTY_IMG_SIZE={args.img_size} in face-service so inference matches."
                )
    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()
