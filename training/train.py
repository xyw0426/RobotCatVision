
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


# ==========================================
# 1. 基础配置
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "fer2013.csv"
OUTPUT_DIR = BASE_DIR / "training" / "outputs"

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001

EARLY_STOP_PATIENCE = 7
SEED = 42

LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# ==========================================
# 2. FER2013 数据集
# ==========================================

class FERDataset(Dataset):
    def __init__(self, dataframe, augment=False):
        self.augment = augment

        # 将 CSV 中的像素字符串转换为图像
        self.images = np.stack([
            np.fromstring(
                pixels,
                dtype=np.uint8,
                sep=" "
            ).reshape(48, 48)
            for pixels in dataframe["pixels"]
        ])

        self.labels = dataframe["emotion"].to_numpy(
            dtype=np.int64
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])

        # 转为 [1, 48, 48]，像素范围 [0, 1]
        image = torch.from_numpy(
            image.copy()
        ).unsqueeze(0).float() / 255.0

        # 仅对训练集做随机水平翻转
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=[2])

        return image, label


# ==========================================
# 3. 单轮训练
# ==========================================

def train_one_epoch(model, loader, criterion,
                    optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total += batch_size

    return total_loss / total, correct / total


# ==========================================
# 4. 验证
# ==========================================

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
        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total += batch_size

    return total_loss / total, correct / total


# ==========================================
# 5. 保存训练曲线
# ==========================================

def save_curves(history):

    epochs = [row[0] for row in history]

    train_loss = [row[1] for row in history]
    train_acc = [row[2] for row in history]

    val_loss = [row[3] for row in history]
    val_acc = [row[4] for row in history]

    # Loss 曲线
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RobotCat EmotionCNN V1 - Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "loss_curve.png",
        dpi=160
    )
    plt.close()

    # Accuracy 曲线
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        [v * 100 for v in train_acc],
        label="Train Accuracy"
    )
    plt.plot(
        epochs,
        [v * 100 for v in val_acc],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("RobotCat EmotionCNN V1 - Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "accuracy_curve.png",
        dpi=160
    )
    plt.close()


# ==========================================
# 6. 主训练函数
# ==========================================

def main():

    # 固定随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # 检测训练设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("RobotCat EmotionCNN V1 Training")
    print("=" * 55)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # 读取 FER2013
    print("\nLoading dataset...", flush=True)

    df = pd.read_csv(DATA_PATH)

    required_columns = {"emotion", "pixels", "Usage"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain: emotion, pixels, Usage"
        )

    train_df = df[df["Usage"] == "Training"]
    val_df = df[df["Usage"] == "PublicTest"]

    if train_df.empty or val_df.empty:
        raise ValueError(
            "Training or PublicTest split is empty."
        )

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # 创建 Dataset
    train_dataset = FERDataset(
        train_df,
        augment=True
    )

    val_dataset = FERDataset(
        val_df,
        augment=False
    )

    # Windows 下采用单进程读取数据
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # 创建神经网络
    model = EmotionCNN(num_classes=7).to(device)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Model parameters: {total_params:,}")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_accuracy = -1.0
    best_val_loss = float("inf")
    no_improve = 0

    history = []

    print("\nStarting training...\n", flush=True)

    # ======================================
    # 7. 训练循环
    # ======================================

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        # 根据验证损失调整学习率
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"LR: {current_lr:.6f}",
            flush=True
        )

        history.append([
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc
        ])

        # 根据验证准确率保存最佳权重
        if val_acc > best_accuracy:

            best_accuracy = val_acc

            torch.save(
                model.state_dict(),
                OUTPUT_DIR / "emotion_cnn_v1.pth"
            )

            print(
                "  [SAVE] New best model!",
                flush=True
            )

        # 根据验证损失判断是否早停
        if val_loss < best_val_loss - 0.0001:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print(
                "\n[EARLY STOP] Validation loss "
                "has not improved.",
                flush=True
            )
            break

    # ======================================
    # 8. 保存训练记录
    # ======================================

    history_path = OUTPUT_DIR / "training_history.csv"

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc"
        ])

        writer.writerows(history)

    save_curves(history)

    print("\n" + "=" * 55)
    print("Training completed!")
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
