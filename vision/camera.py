import platform
import cv2


def open_camera(camera_id=0, width=640, height=480):
    """
    跨平台摄像头初始化。

    Windows: DirectShow
    macOS: AVFoundation
    Linux: V4L2 (适合普通 USB 摄像头)

    树莓派 CSI 摄像头可能需要 Picamera2 适配。
    """

    system = platform.system()

    if system == "Windows":
        backend = cv2.CAP_DSHOW
    elif system == "Darwin":
        backend = cv2.CAP_AVFOUNDATION
    else:
        backend = cv2.CAP_V4L2

    camera = cv2.VideoCapture(camera_id, backend)

    # 失败时尝试 OpenCV 默认后端
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_id)

    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id} on {system}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"[INFO] Camera opened on {system}, " f"backend={camera.getBackendName()}")

    return camera
