import os
import csv
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data

# File structure:
# - model/
#   - celeba/
#       - img_align_celeba/
#           [images]
#       - list_attr_celeba.csv
#       - list_bbox_celeba.csv
#       - list_eval_partition.csv
#       - list_landmarks_align_celeba.csv


DEVICE = "mps" # or cpu or cuda

IMG_DIR = "./celeba/img_align_celeba"
ATTR_CSV = "./celeba/list_attr_celeba.csv"

NUM_IMAGES = 10000
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 10
LR = 1e-3
TRAIN_RATIO = 0.75
IMG_SIZE = 128
NUM_ATTRS = 40


class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_csv, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        self.attr_names = []

        with open(attr_csv, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.attr_names = header[1:]  # skip image_id column
            for row in reader:
                filename = row[0]
                # remap -1/1 -> 0/1
                attrs = [(int(v) + 1) // 2 for v in row[1:]]
                self.samples.append((filename, attrs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, attrs = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(attrs, dtype=torch.float32)
        return img, label


class CelebACNN(nn.Module):
    def __init__(self, num_attrs=NUM_ATTRS):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3x128x128 -> 32x64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x64x64 -> 64x32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3: 64x32x32 -> 128x16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 4: 128x16x16 -> 256x8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            # Block 5: 256x8x8 -> 512x4x4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_attrs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        if (batch_idx + 1) % 200 == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)}, loss: {loss.item():.4f}")

    return running_loss / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_preds = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total_preds += labels.numel()

    avg_loss = running_loss / total_samples
    accuracy = correct / total_preds
    return avg_loss, accuracy


def main():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    dataset = CelebADataset(IMG_DIR, ATTR_CSV, transform=transform)

    if NUM_IMAGES < len(dataset):
        dataset, _ = random_split(
            dataset, [NUM_IMAGES, len(dataset) - NUM_IMAGES],
            generator=torch.Generator().manual_seed(42),
        )

    train_size = int(len(dataset) * TRAIN_RATIO)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    model = CelebACNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nTraining...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} — "
            f"train loss: {train_loss:.4f}, "
            f"test loss: {test_loss:.4f}, "
            f"test acc: {test_acc:.4f}"
        )

    save_path = os.path.join(os.path.dirname(__file__), "celeba_cnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
