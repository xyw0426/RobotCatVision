from pathlib import Path
import random
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import EmotionCNN

# ======================================
# 1. 训练配置
# ======================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "fer2013.csv"

OUTPUT_DIR = BASE_DIR / "training" / "outputs" / "cross_platform_test"

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 7
SEED = 42


# ======================================
# 2. 跨平台计算设备
# ======================================


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ======================================
# 3. FER2013 数据集
# ======================================


class FERDataset(Dataset):
    def __init__(self, dataframe, augment=False):
        self.augment = augment

        self.images = np.stack(
            [
                np.fromstring(pixels, dtype=np.uint8, sep=" ").reshape(48, 48)
                for pixels in dataframe["pixels"]
            ]
        )

        self.labels = dataframe["emotion"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy()).unsqueeze(0).float() / 255.0

        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=[2])

        return image, int(self.labels[index])


# ======================================
# 4. 单轮训练
# ======================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


# ======================================
# 5. 验证
# ======================================


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


# ======================================
# 6. 绘制训练曲线
# ======================================


def save_curves(history):
    epochs = [row[0] for row in history]

    train_loss = [row[1] for row in history]
    train_acc = [row[2] * 100 for row in history]
    val_loss = [row[3] for row in history]
    val_acc = [row[4] * 100 for row in history]

    for name, train_values, val_values, ylabel in [
        ("loss_curve.png", train_loss, val_loss, "Loss"),
        ("accuracy_curve.png", train_acc, val_acc, "Accuracy (%)"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_values, label="Train")
        plt.plot(epochs, val_values, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"RobotCat - {ylabel}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / name, dpi=160)
        plt.close()


# ======================================
# 7. 主训练程序
# ======================================


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    device = get_device()

    print("=" * 55)
    print("RobotCat Cross-platform CNN Training")
    print("=" * 55)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    elif device.type == "mps":
        print("GPU: Apple Metal Performance Shaders")
    else:
        print("CPU training mode")

    # 加载数据
    df = pd.read_csv(DATA_PATH)

    required = {"emotion", "pixels", "Usage"}

    if not required.issubset(df.columns):
        raise ValueError("CSV requires emotion, pixels, Usage")

    train_df = df[df["Usage"] == "Training"]
    val_df = df[df["Usage"] == "PublicTest"]

    if train_df.empty or val_df.empty:
        raise ValueError("Training/validation data missing")

    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))

    train_dataset = FERDataset(train_df, augment=True)
    val_dataset = FERDataset(val_df, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = EmotionCNN(num_classes=7).to(device)

    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_accuracy = -1.0
    best_val_loss = float("inf")
    no_improve = 0
    history = []

    print("\nStarting training...", flush=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"LR: {current_lr:.6f}",
            flush=True,
        )

        history.append([epoch, train_loss, train_acc, val_loss, val_acc])

        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc

            torch.save(model.state_dict(), OUTPUT_DIR / "emotion_cnn.pth")

            print("[SAVE] Best model updated")

        # 早停
        if val_loss < best_val_loss - 0.0001:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print("[EARLY STOP] Training stopped")
            break

    # 保存训练历史
    with open(
        OUTPUT_DIR / "training_history.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        writer.writerows(history)

    save_curves(history)

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print("Results:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
