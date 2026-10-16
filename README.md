# RobotCat

面向机器猫的人脸追踪、七类面部表情识别和模拟头部控制原型。

程序通过摄像头获取画面，用 MediaPipe 检测人脸，持续跟踪一个目标，并使用 PyTorch 轻量 CNN 识别人脸表情。它根据目标偏离画面中心的位置计算左右转动（Yaw）和上下俯仰（Pitch）角度。

当前头部控制只计算虚拟角度，尚未接入真实舵机。树莓派是部署方向，硬件适配与性能仍需实机验证。

## 当前功能

- 实时摄像头画面和人脸检测。
- 单目标人脸追踪、位置平滑和丢失处理。
- 七类表情：Angry、Disgust、Fear、Happy、Sad、Surprise、Neutral。
- 表情概率平滑及低置信度提示。
- 带死区、速度和角度限制的模拟头部控制。
- FER2013 训练、独立测试集评估和训练曲线生成。

## 项目结构

```text
RobotCat/
├── main.py                         # 实时视觉程序入口
├── vision/
│   ├── camera.py                   # 跨平台摄像头初始化
│   ├── face_tracker.py             # 人脸目标追踪
│   └── emotion_classifier.py       # 人脸预处理和表情推理
├── control/head_controller.py      # 虚拟头部角度控制
├── training/
│   ├── model.py                    # EmotionCNN 网络定义
│   ├── train.py                    # CUDA / CPU 训练
│   ├── train_cross_platform.py     # CUDA / MPS / CPU 训练
│   ├── evaluate.py                 # PrivateTest 评估
│   └── outputs/                    # 基线模型、曲线和训练记录
├── models/face_detector.tflite      # MediaPipe 人脸检测模型
├── data/                           # 本地数据集，CSV 不提交到 Git
├── requirements.txt
└── CONTRIBUTING.md
```

## 安装与运行

以下命令在项目根目录执行。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

macOS / Linux 可将上述 `.\.venv\Scripts\python.exe` 替换为 `.venv/bin/python`，创建环境时按本机情况使用 `python3`。

依赖清单尚未锁定版本，也未完成各平台安装验证。GPU 训练时，需要安装与本机驱动匹配的 PyTorch。树莓派上的依赖安装和推理性能需要单独验证。

运行前需要连接可用摄像头，并确保以下模型文件存在：

- `models/face_detector.tflite`
- `training/outputs/emotion_cnn_v1.pth`

程序默认使用编号为 `0` 的摄像头，可在 `main.py` 中修改 `CAMERA_ID`。表情推理默认使用 CPU。按 **Q** 退出，按 **R** 将虚拟头部角度归零。

运行实时演示不需要 FER2013 数据集。树莓派 CSI 摄像头可能需要 Picamera2 适配，目前摄像头模块按普通 USB 摄像头方式访问 Linux 摄像头。

## 训练与评估

训练前请按 [数据准备说明](data/README.md) 将 FER2013 CSV 放到 `data/fer2013.csv`。

```powershell
.\.venv\Scripts\python.exe training/train.py
.\.venv\Scripts\python.exe training/evaluate.py
```

跨平台训练脚本：

```powershell
.\.venv\Scripts\python.exe training/train_cross_platform.py
```

常规训练将覆盖 `training/outputs/` 中同名的模型、训练记录和曲线，实验前请保存需要保留的基线结果。跨平台脚本写入 `training/outputs/cross_platform_test/`，该目录默认不提交到 Git；主程序仍加载常规训练目录中的模型。

数据划分：`Training` 用于训练，`PublicTest` 用于验证，`PrivateTest` 用于独立测试。网络输入为归一化到 `[0, 1]` 的 `48×48` 单通道图像。

现有训练记录包含 30 轮训练，最高验证准确率为 **57.62%（第 29 轮）**。这是数据集验证结果，不代表摄像头实测准确率；仓库中未保存独立测试准确率的文字报告。

## 协作

请阅读 [协作说明](CONTRIBUTING.md)。建议通过 Issue 描述问题、在功能分支修改、提交 Pull Request 后由队友检查并合并。

后续可协作的方向包括：树莓派部署、真实舵机接入、摄像头场景识别效果改进、依赖版本固定和运行环境验证。
