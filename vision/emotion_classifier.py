from pathlib import Path

import cv2
import numpy as np
import torch

from training.model import EmotionCNN

# ==========================================
# 1. 表情类别
# ==========================================

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# ==========================================
# 2. 表情识别器
# ==========================================


class EmotionClassifier:

    def __init__(
        self,
        model_path,
        device="cpu",
        crop_mode="margin",
        crop_scale=1.3,
        min_face_size=32,
    ):
        """
        model_path: CNN 权重路径
        device: cpu / cuda
        crop_mode: original / margin
        crop_scale: 扩大裁剪框的倍数
        min_face_size: 最小有效人脸尺寸
        """

        self.device = torch.device(device)

        self.crop_mode = crop_mode
        self.crop_scale = crop_scale
        self.min_face_size = min_face_size

        if self.crop_mode not in ("original", "margin"):
            raise ValueError("crop_mode must be 'original' or 'margin'")

        if self.crop_scale < 1.0:
            raise ValueError("crop_scale must be >= 1.0")

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Emotion model not found: {model_path}")

        # 加载 CNN 网络
        self.model = EmotionCNN(num_classes=7).to(self.device)

        weights = torch.load(model_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(weights)
        self.model.eval()

        print(f"[INFO] EmotionCNN loaded on {self.device}")
        print(f"[INFO] Crop mode: {self.crop_mode}, " f"scale: {self.crop_scale}")

    # ======================================
    # 3. 计算裁剪区域
    # ======================================

    def get_crop_box(self, frame, box):
        """
        输入 MediaPipe 的 (x, y, w, h)。
        返回经过裁剪模式调整后的安全坐标：
        (x1, y1, x2, y2)
        """

        x, y, w, h = box

        frame_h, frame_w = frame.shape[:2]

        if w <= 0 or h <= 0:
            return None

        # 使用原始检测框
        if self.crop_mode == "original":
            scale = 1.0
        else:
            scale = self.crop_scale

        # 检测框中心
        cx = x + w / 2.0
        cy = y + h / 2.0

        # 扩大裁剪框
        new_w = w * scale
        new_h = h * scale

        # 转换为裁剪坐标
        x1 = int(round(cx - new_w / 2))
        y1 = int(round(cy - new_h / 2))
        x2 = int(round(cx + new_w / 2))
        y2 = int(round(cy + new_h / 2))

        # 限制在摄像头画面内部
        x1 = max(0, min(frame_w, x1))
        y1 = max(0, min(frame_h, y1))
        x2 = max(0, min(frame_w, x2))
        y2 = max(0, min(frame_h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    # ======================================
    # 4. 人脸预处理
    # ======================================

    def preprocess(self, frame, box):
        """
        输出:
            Tensor [1, 1, 48, 48]
            或 None
        """

        x, y, w, h = box

        # 判断原始检测框是否过小
        if w < self.min_face_size or h < self.min_face_size:
            return None

        crop_box = self.get_crop_box(frame, box)

        if crop_box is None:
            return None

        x1, y1, x2, y2 = crop_box

        # 裁剪人脸
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            return None

        # BGR 转灰度
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # 根据实际尺寸选择插值方式
        if gray.shape[0] >= 48 and gray.shape[1] >= 48:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        # 缩放到 CNN 固定输入尺寸
        resized = cv2.resize(gray, (48, 48), interpolation=interpolation)

        # 归一化：[0, 255] -> [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # 转换为 [1, 1, 48, 48]
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        return tensor.to(self.device)

    # ======================================
    # 5. CNN 推理
    # ======================================

    @torch.inference_mode()
    def predict(self, frame, box):
        """
        返回:
        {
            "label": "Neutral",
            "confidence": 0.54,
            "probabilities": numpy.ndarray
        }

        若人脸区域无效，则返回 None。
        """

        image = self.preprocess(frame, box)

        if image is None:
            return None

        # CNN 前向传播
        logits = self.model(image)

        # 转换为 Softmax 分数
        probabilities = torch.softmax(logits, dim=1)[0]

        index = int(probabilities.argmax().item())

        confidence = float(probabilities[index].item())

        return {
            "label": LABELS[index],
            "confidence": confidence,
            "probabilities": (probabilities.cpu().numpy()),
        }
