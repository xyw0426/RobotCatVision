from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from model import EmotionCNN
from train import FERDataset

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "fer2013.csv"
MODEL_PATH = BASE_DIR / "training" / "outputs" / "emotion_cnn_v1.pth"

OUTPUT_DIR = BASE_DIR / "training" / "outputs"

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # 读取独立测试集
    df = pd.read_csv(DATA_PATH)

    test_df = df[df["Usage"] == "PrivateTest"]

    if test_df.empty:
        raise ValueError("PrivateTest dataset is empty")

    print("Test samples:", len(test_df))

    test_dataset = FERDataset(test_df, augment=False)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # 加载训练好的模型
    model = EmotionCNN(num_classes=7).to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )

    model.eval()

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nTest Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(7)),
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    # 绘制归一化混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)), normalize="true")

    fig, ax = plt.subplots(figsize=(9, 7))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)

    disp.plot(
        ax=ax, cmap="Blues", values_format=".2f", xticks_rotation=45, colorbar=True
    )

    ax.set_title("RobotCat EmotionCNN V1 - Confusion Matrix")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=180)
    plt.close()

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
