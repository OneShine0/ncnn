# Raspberry Pi Deployment Tutorial / 树莓派傻瓜式部署教程

这份文档按“先用 Python 跑通”的路线写。目标是让一台干净的 Raspberry Pi OS 可以运行当前 YOLOv5 NCNN 检测程序。C++ NCNN 路线放在最后，作为 `pip install ncnn` 失败或性能不足时的兜底方案。

This guide uses the Python-first route. The goal is to run the detector on a clean Raspberry Pi OS. The C++ NCNN route is kept as a fallback.

## 1. 准备树莓派 / Prepare the Pi

建议使用 Raspberry Pi OS 64-bit，并先更新系统：

```bash
sudo apt update
sudo apt upgrade -y
```

安装基础工具和 OpenCV：

```bash
sudo apt install -y python3 python3-pip python3-venv python3-opencv git v4l-utils
```

检查 Python 和摄像头工具：

```bash
python3 --version
v4l2-ctl --list-devices
```

If `v4l2-ctl` lists your USB camera, the system can see the camera device.

## 2. 拷贝项目文件 / Copy Project Files

从 Windows 或 GitHub 把项目放到树莓派，例如：

```bash
cd ~
git clone <你的 GitHub 仓库地址> ncnn
cd ncnn
```

如果不用 Git，也可以手动复制这些必要文件：

```text
best.ncnn.param
best.ncnn.bin
labels.txt
detect_ncnn_yolov5.py
camera_input.py
roi_motion.py
backend_client.py
requirements.txt
README.md
RASPBERRY_PI_DEPLOYMENT.md
docs/
```

不要复制这些 Windows 本地目录：

```text
.venv/
python_packages/
__pycache__/
.tmp/
.vscode/
```

`python_packages/` contains Windows-only dependency files and is not suitable for Raspberry Pi.

## 3. 安装 Python 依赖 / Install Python Dependencies

先尝试用户级安装：

```bash
python3 -m pip install --user ncnn numpy
```

OpenCV 已通过系统包 `python3-opencv` 安装，一般不需要再 pip 安装 `opencv-python`。

如果系统提示 externally managed environment，可以使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ncnn numpy
```

如果使用虚拟环境，后续命令请先运行：

```bash
source .venv/bin/activate
```

## 4. 环境自检 / Self Test

进入项目目录：

```bash
cd ~/ncnn
```

运行自检：

```bash
python3 detect_ncnn_yolov5.py --self-test
```

看到类似输出就说明模型和 Python 环境基本可用：

```text
environment: ok
model: best.ncnn.param + best.ncnn.bin
input: in0, output: out0, size: 320
blank-image detections: 0
```

If self-test passes, the NCNN runtime and model files are usable.

## 5. USB 摄像头运行 / Run with USB Camera

先确认摄像头编号，通常是 `/dev/video0`，对应 `--camera-index 0`：

```bash
v4l2-ctl --list-devices
```

有桌面环境时可以打开预览窗口：

```bash
python3 detect_ncnn_yolov5.py --camera-index 0 --camera-width 640 --camera-height 640 --threads 4
```

无桌面或 SSH 环境请使用 headless：

```bash
python3 detect_ncnn_yolov5.py \
  --headless \
  --camera-index 0 \
  --camera-width 640 \
  --camera-height 640 \
  --threads 4
```

Headless mode is recommended on Raspberry Pi because GUI preview costs CPU and may fail over SSH.

## 6. Raspberry Pi Camera Module / 树莓派 CSI 摄像头

如果使用树莓派摄像头模块，优先确认系统能看到摄像头：

```bash
libcamera-hello
```

当前代码默认使用 OpenCV `VideoCapture`。如果 CSI 摄像头不能通过 `/dev/video0` 打开，需要在 `camera_input.py` 中把读取实现替换为 Picamera2，但保持这个接口不变：

```python
def read_frame(self) -> tuple[bool, np.ndarray | None]:
    ...
```

返回帧必须是 BGR、`uint8`、`height x width x 3`。

For Picamera2, return BGR `uint8` frames and keep the same public methods.

## 7. 后端上报 / Backend Reporting

如果要把检测结果发给后端：

```bash
python3 detect_ncnn_yolov5.py \
  --headless \
  --backend-url http://HOST:PORT/api/detections \
  --device-id pi-camera-01 \
  --camera-index 0 \
  --camera-width 640 \
  --camera-height 640 \
  --threads 4
```

后端会收到 HTTP `POST` JSON：

```json
{
  "device_id": "pi-camera-01",
  "timestamp_ms": 1777890000000,
  "frame_id": 128,
  "image_size": {"width": 640, "height": 640},
  "roi": {"x1": 112, "y1": 80, "x2": 420, "y2": 380, "width": 308, "height": 300, "reason": "motion"},
  "fps": 18.42,
  "inference_ms": 12.37,
  "detections": [
    {
      "class_id": 0,
      "label": "Correct Wear",
      "confidence": 0.9345,
      "cached": false,
      "box": {"x1": 160, "y1": 96, "x2": 236, "y2": 210}
    }
  ]
}
```

Any `2xx` backend response is accepted.

## 8. 推荐参数 / Recommended Pi Parameters

先使用默认参数，只明确写这些：

```bash
python3 detect_ncnn_yolov5.py \
  --headless \
  --camera-index 0 \
  --camera-width 640 \
  --camera-height 640 \
  --threads 4 \
  --roi-mode hybrid \
  --full-frame-refresh-mode tiles \
  --full-frame-refresh-ms 2000 \
  --roi-padding-pixels 50 \
  --roi-smooth-alpha 0.6 \
  --face-refresh-ms 500 \
  --detection-ttl-ms 500 \
  --show-expired-ttl \
  --expired-ttl-display-frames 8 \
  --frame-diff-enabled \
  --frame-diff-threshold 12 \
  --frame-diff-min-area 300 \
  --frame-diff-alpha 0.08 \
  --status-overlay compact
```

Default refresh behavior checks three tiles over each refresh cycle: the left half, the right half, and one centered vertical patch. On 640x640 frames the centered patch is `[160,0,480,640]`. This avoids shrinking the whole frame on every periodic refresh while keeping Raspberry Pi load to at most one inference per frame. Non-full ROI detections are regional evidence: stale boxes inside the checked ROI are removed if they are not detected again. During testing, TTL-expired boxes are briefly drawn as red dashed `TTL expired` ghosts. Frame-diff motion is enabled by default as a lightweight backup for subtle movement; lower threshold values are more sensitive but noisier.

调参顺序：

- 检测框消失太慢：降低 `--full-frame-refresh-ms`。
- 人脸被 ROI 裁掉：增大 `--roi-padding-pixels`。
- 小动作检测不够及时：降低 `--face-refresh-ms`。
- 光照变化导致误触发：增大 `--roi-min-area` 或 `--mog2-var-threshold`。
- FPS 不够：保持 `--headless`，尝试 `--threads 2`、`3`、`4` 比较实际效果。

## 9. 常见问题 / Troubleshooting

### 找不到 ncnn / `ModuleNotFoundError: No module named 'ncnn'`

重新安装：

```bash
python3 -m pip install --user ncnn numpy
```

如果使用虚拟环境：

```bash
source .venv/bin/activate
python -m pip install ncnn numpy
```

### 摄像头打不开 / Could not open camera index 0

检查设备：

```bash
v4l2-ctl --list-devices
ls /dev/video*
```

尝试其他编号：

```bash
python3 detect_ncnn_yolov5.py --camera-index 1 --headless
```

### OpenCV 窗口失败 / GUI preview fails

SSH 或无桌面环境请加：

```bash
--headless
```

### FPS 太低 / Low FPS

- 使用 `--headless`。
- 保持输入尺寸 `640x640`，模型尺寸默认 `320x320`。
- 优先使用 `--roi-mode hybrid`。
- 适当增大 `--face-refresh-ms`。
- 尝试不同 `--threads`。

### pip 没有树莓派 ncnn wheel

如果 `python3 -m pip install ncnn` 找不到适合当前系统和 Python 版本的 wheel，就走下面的 C++ fallback。

## 10. C++ NCNN Fallback

这条路线适合 Python `ncnn` 安装失败，或后续需要更高性能时使用。

安装编译工具：

```bash
sudo apt update
sudo apt install -y build-essential git cmake libopencv-dev
```

编译 NCNN：

```bash
cd ~
git clone --depth=1 https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
mkdir -p build
cd build
cmake -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON ..
cmake --build . -j$(nproc)
sudo cmake --install .
```

模型约定保持不变：

```text
Input blob: in0
Output blob: out0
Input size: 320x320
Preprocess: BGR to RGB, letterbox value 114, normalize by 1/255
Output row: x, y, w, h, obj, class0, class1, class2
Final score: obj * class_score
```
