from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

from vision.camera import open_camera
from vision.face_tracker import FaceTracker
from vision.emotion_classifier import EmotionClassifier, LABELS
from control.head_controller import HeadController

# ======================================
# 1. 项目配置
# ======================================

BASE_DIR = Path(__file__).resolve().parent

FACE_MODEL = BASE_DIR / "models" / "face_detector.tflite"

EMOTION_MODEL = BASE_DIR / "training" / "outputs" / "emotion_cnn_v1.pth"

CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 追踪参数
TRACK_ALPHA = 0.35
MAX_MISSING_FRAMES = 10

# 头部控制参数：仅用于模拟
YAW_LIMIT = (-60, 60)
PITCH_LIMIT = (-20, 30)
CONTROL_GAIN = 45.0
CONTROL_DEADZONE = 0.06
MAX_HEAD_SPEED = 60.0

# 表情分类参数
EMOTION_INTERVAL = 3
EMOTION_ALPHA = 0.3
CONFIDENCE_THRESHOLD = 0.45

WINDOW_NAME = "RobotCat Vision System"


# ======================================
# 2. 工具函数
# ======================================


def draw_text(frame, text, position, color=(255, 255, 255)):
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA
    )


# ======================================
# 3. 主程序
# ======================================


def main():
    print("[INFO] Starting RobotCat...", flush=True)

    for model_path in (FACE_MODEL, EMOTION_MODEL):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

    tracker = FaceTracker(alpha=TRACK_ALPHA, max_missing=MAX_MISSING_FRAMES)

    head = HeadController(
        yaw_limit=YAW_LIMIT,
        pitch_limit=PITCH_LIMIT,
        gain=CONTROL_GAIN,
        deadzone=CONTROL_DEADZONE,
        max_speed=MAX_HEAD_SPEED,
    )

    # 轻量 CNN 默认使用 CPU 推理
    emotion = EmotionClassifier(
        model_path=EMOTION_MODEL, device="cpu", crop_mode="margin", crop_scale=1.3
    )

    # MediaPipe Face Detector
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )

    camera = open_camera(camera_id=CAMERA_ID, width=FRAME_WIDTH, height=FRAME_HEIGHT)

    start_time = time.monotonic()
    previous_time = time.perf_counter()
    last_timestamp = -1

    fps = 0.0
    frame_count = 0

    emotion_probs = None

    print("[INFO] Q: Quit | R: Reset head")

    try:
        with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:

            while True:
                success, frame = camera.read()

                if not success:
                    print("[WARNING] Camera frame unavailable.")
                    break

                raw_frame = frame.copy()
                height, width = frame.shape[:2]

                # -------------------------------
                # A. 帧率
                # -------------------------------

                now = time.perf_counter()
                dt = now - previous_time
                previous_time = now

                if dt > 0:
                    instant_fps = 1.0 / dt
                    fps = instant_fps if fps == 0 else 0.8 * fps + 0.2 * instant_fps

                frame_count += 1

                # -------------------------------
                # B. 人脸检测
                # -------------------------------

                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp = int((time.monotonic() - start_time) * 1000)

                timestamp = max(timestamp, last_timestamp + 1)
                last_timestamp = timestamp

                result = detector.detect_for_video(mp_image, timestamp)

                detections = result.detections or []

                # -------------------------------
                # C. 目标追踪
                # -------------------------------

                was_active = tracker.active
                target_center = tracker.update(detections)

                # 目标已被清除：清除旧的表情状态
                if was_active and not tracker.active:
                    emotion_probs = None

                # 新目标开始追踪
                if not was_active and tracker.active:
                    emotion_probs = None

                # 普通人脸：蓝框
                for face in detections:
                    b = face.bounding_box

                    cv2.rectangle(
                        frame,
                        (b.origin_x, b.origin_y),
                        (b.origin_x + b.width, b.origin_y + b.height),
                        (255, 0, 0),
                        2,
                    )

                # -------------------------------
                # D. 表情识别及头部控制
                # -------------------------------

                expression_text = "No face"

                if target_center is not None:
                    x, y, w, h = tracker.box
                    cx, cy = map(int, target_center)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                    # 间隔若干帧执行一次 CNN
                    if frame_count % EMOTION_INTERVAL == 0:
                        prediction = emotion.predict(raw_frame, tracker.box)

                        if prediction is not None:
                            new_probs = prediction["probabilities"]

                            if emotion_probs is None:
                                emotion_probs = new_probs.copy()
                            else:
                                emotion_probs = (
                                    EMOTION_ALPHA * new_probs
                                    + (1 - EMOTION_ALPHA) * emotion_probs
                                )

                    # 输出平滑后的表情
                    if emotion_probs is not None:
                        index = int(np.argmax(emotion_probs))
                        confidence = float(emotion_probs[index])

                        if confidence >= CONFIDENCE_THRESHOLD:
                            expression_text = f"{LABELS[index]} {confidence:.0%}"
                        else:
                            expression_text = (
                                f"Uncertain ({LABELS[index]} " f"{confidence:.0%})"
                            )
                    else:
                        expression_text = "Analyzing..."

                    # 归一化追踪误差
                    error_x = (cx - width / 2) / (width / 2)

                    error_y = (cy - height / 2) / (height / 2)

                    yaw, pitch = head.update(error_x, error_y, dt)

                    cv2.line(
                        frame, (width // 2, height // 2), (cx, cy), (0, 255, 255), 1
                    )

                    status = "TRACKING"

                else:
                    # 目标暂时丢失：停止更新表情
                    emotion_probs = None
                    yaw = head.yaw
                    pitch = head.pitch

                    status = "TARGET LOST" if tracker.active else "SEARCHING"

                # -------------------------------
                # E. 画面可视化
                # -------------------------------

                cv2.drawMarker(
                    frame,
                    (width // 2, height // 2),
                    (0, 255, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )

                cv2.rectangle(frame, (0, 0), (width, 155), (30, 30, 30), -1)

                draw_text(frame, f"FPS: {fps:.1f}   Faces: {len(detections)}", (10, 25))

                draw_text(
                    frame,
                    f"Status: {status}",
                    (10, 53),
                    (0, 255, 0) if status == "TRACKING" else (0, 165, 255),
                )

                draw_text(frame, f"Emotion: {expression_text}", (10, 81), (0, 255, 255))

                draw_text(frame, f"Yaw: {yaw:+.1f} deg", (10, 109))

                draw_text(frame, f"Pitch: {pitch:+.1f} deg", (10, 137))

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("r"):
                    head.yaw = 0.0
                    head.pitch = 0.0
                    print("[INFO] Virtual head reset.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] RobotCat stopped.")


if __name__ == "__main__":
    main()
