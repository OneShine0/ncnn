# YOLOv5 NCNN Camera Detector

本项目用于运行导出的 YOLOv5 NCNN 模型，支持图片、视频文件和摄像头实时检测。当前类别用于口罩佩戴状态识别：

- `0`: Correct Wear，绿色框
- `1`: Wrong Wear，黄色/橙色框
- `2`: No Wear，红色框

This project runs an exported YOLOv5 NCNN model for image, video-file, and live-camera inference. It detects mask-wearing status with three classes.

## 文件结构 / Project Layout

```text
best.ncnn.param              # NCNN model structure
best.ncnn.bin                # NCNN model weights
labels.txt                   # Class labels
detect_ncnn_yolov5.py        # Main entry point
camera_input.py              # Camera source wrapper
roi_motion.py                # Motion ROI selector
backend_client.py            # Async backend JSON sender
raw_stream.py                # Raw MJPEG stream server for backend preview
requirements.txt             # Python dependency list
RASPBERRY_PI_DEPLOYMENT.md   # Raspberry Pi beginner tutorial
docs/                        # Bilingual module notes
videos/                      # Local test videos, not required on Raspberry Pi
python_packages/             # Local Windows dependency copy, do not copy to Raspberry Pi
```

`python_packages/` 是 Windows 本机运行时的依赖副本，里面包含 Windows 专用文件，例如 `pywin32`、`win32com` 和 `*-win_amd64.pyd`。它保留在本地使用，并被 `.gitignore` 排除；树莓派部署请按 `RASPBERRY_PI_DEPLOYMENT.md` 安装依赖，不要直接复制这个目录。

`python_packages/` is a local Windows dependency copy. It contains Windows-only files, so keep it local and ignored by Git. Do not copy it to Raspberry Pi.

## Windows 环境 / Windows Setup

在项目目录下运行：

```powershell
cd C:\Users\71549\Desktop\ncnn
python .\detect_ncnn_yolov5.py --self-test
```

如果提示找不到 `ncnn`，可使用当前项目的本地依赖目录，或重新安装：

```powershell
python -m pip install --target .\python_packages ncnn opencv-python numpy
```

If `ncnn` is missing on Windows, install it into the local package folder shown above.

## 自检 / Self Test

```powershell
python .\detect_ncnn_yolov5.py --self-test
```

成功时会打印模型名、输入输出 blob 名、类别和一次空白图推理耗时。

On success, the script prints model names, blob names, labels, and blank-frame inference time.

## 图片检测 / Image Inference

```powershell
python .\detect_ncnn_yolov5.py --image .\test.jpg --output .\result.jpg
```

如果不写 `--output`，程序会在原图同目录生成 `*_ncnn` 后缀的图片。

If `--output` is omitted, the script writes an image with a `*_ncnn` suffix next to the input file.

## 视频检测 / Video Inference

你已经在 `videos/` 文件夹准备了 3 个视频，可以直接运行：

```powershell
python .\detect_ncnn_yolov5.py --video .\videos\video1.mp4 --output .\videos\video1_detected.mp4
python .\detect_ncnn_yolov5.py --video .\videos\video2.mp4 --output .\videos\video2_detected.mp4
python .\detect_ncnn_yolov5.py --video .\videos\video3.mp4 --output .\videos\video3_detected.mp4
```

不想弹出预览窗口时，加 `--headless`：

```powershell
python .\detect_ncnn_yolov5.py --video .\videos\video1.mp4 --output .\videos\video1_detected.mp4 --headless
```

Video mode reads frames from `--video`, draws detections, and writes a rendered video when `--output` is provided. `--headless` disables the preview window.

## 摄像头检测 / Camera Inference

默认打开 0 号摄像头：

```powershell
python .\detect_ncnn_yolov5.py
```

后台运行并上报后端：

```powershell
python .\detect_ncnn_yolov5.py --headless --backend-url http://HOST:PORT/api/detections --device-id win-camera-01
```

Default mode opens camera index `0`. Use `--headless` for no preview window and `--backend-url` to post JSON detections.

## PC 后端实时预览 / PC Backend Live Preview

当前推荐链路是：树莓派运行 `mjpg-streamer` 输出原始 MJPEG，PC 检测端读取这个网络流做 NCNN 检测，并把 JSON 上报给本机后端：

```powershell
python .\detect_ncnn_yolov5.py `
  --stream-url "http://172.20.10.2:8080/?action=stream" `
  --backend-url http://127.0.0.1:8000/api/detections `
  --device-id pi-camera-01
```

后端设备配置：

```text
pi-camera-01 -> http://172.20.10.2:8080/?action=stream
```

如果你的 `mjpg-streamer` 首页地址本身就是裸流，也可以把 `--stream-url` 和后端配置写成：

```text
http://172.20.10.2:8080/
```

视频帧不进入 JSON。后端使用 `mjpg-streamer` 原始 MJPEG + 最新 JSON 近实时叠加，完整需求见 [docs/backend_requirements.md](docs/backend_requirements.md)。项目内的 `--raw-stream` 是备用方案：没有外部推流程序时，检测端才需要自己提供原始 MJPEG。`mjpg-streamer` 通常更流畅，是因为它能直接利用摄像头原生 MJPEG、减少重复 JPEG 编码和图像拷贝，并由专用 C 程序负责推流。

## 常用参数 / Useful Options

```text
--param best.ncnn.param
--bin best.ncnn.bin
--labels labels.txt
--image path
--video path
--stream-url http://172.20.10.2:8080/?action=stream
--output path
--camera-index 0
--camera-width 640
--camera-height 640
--camera-fps 30
--headless
--img-size 320
--conf-thres 0.25
--iou-thres 0.45
--nms-mode class_agnostic
--threads 4
--roi-mode hybrid
--full-frame-refresh-mode tiles
--full-frame-refresh-ms 1000
--roi-min-area 800
--roi-min-size-pixels 96
--roi-padding-pixels 50
--roi-smooth-alpha 0.6
--face-refresh-ms 500
--detection-ttl-ms 0
--show-expired-ttl
--expired-ttl-display-frames 8
--motion-source mog2
--mog2-history 80
--frame-diff-enabled
--frame-diff-threshold 12
--frame-diff-min-area 300
--frame-diff-alpha 0.08
--raw-stream
--raw-stream-host 0.0.0.0
--raw-stream-port 8090
--raw-stream-fps 10
--raw-stream-width 640
--raw-stream-quality 70
--status-overlay compact
--backend-url http://HOST:PORT/api/detections
--device-id pi-camera-01
```

## ROI 策略 / ROI Strategy

- `hybrid`：默认。第一帧和定时刷新做全图检测，其余时间优先使用运动 ROI 和缓存人脸 ROI。
- `roi`：第一帧全图，之后主要依赖运动 ROI。
- `full`：每帧全图检测，最稳定但最慢。

The default `hybrid` mode balances stability and speed. It uses MOG2 motion ROIs, cached-face rechecks, regional positive/negative evidence, and 1000 ms time-sliced refresh tiles. In tile mode the refresh ROIs are the left half, right half, and one centered vertical patch; on 640x640 frames the centered patch is `[160,0,480,640]`. Non-full ROIs are regional evidence: old boxes whose centers fall inside the checked ROI are cleared first, then the new ROI detections are added. `--detection-ttl-ms 0` disables time-based stale-box expiry by default, so ghost boxes are cleaned by ROI negative evidence instead; set a positive value such as `500` to test TTL expiry and the red dashed `TTL expired` overlay. `--status-overlay compact` uses a vertical semi-transparent test panel. `--motion-source mog2` is the recommended default; use `--motion-source frame_diff` for frame-diff-only testing or `--motion-source both` to merge MOG2 and frame diff. `--roi-smooth-alpha 0.6` follows the older stable ROI feel; higher values are steadier but trail movement more, while lower values are more responsive but can jitter. For frame diff, lower `--frame-diff-threshold` is more sensitive but noisier, and lower `--frame-diff-min-area` catches smaller changes but can false-trigger more easily.

## 树莓派快速入口 / Raspberry Pi Quick Link

树莓派新手部署请看：

[RASPBERRY_PI_DEPLOYMENT.md](RASPBERRY_PI_DEPLOYMENT.md)

Beginner-friendly Raspberry Pi deployment is documented in the file above.

## 每个 Python 文件的说明 / Module Documentation

- [detect_ncnn_yolov5.py](docs/detect_ncnn_yolov5.md)
- [camera_input.py](docs/camera_input.md)
- [roi_motion.py](docs/roi_motion.md)
- [backend_client.py](docs/backend_client.md)
- [raw_stream.py](docs/raw_stream.md)
- [backend_requirements.md](docs/backend_requirements.md)

每份文档都包含中文和英文说明，介绍模块职责、主要类/函数、输入输出和移植注意事项。

Each document includes Chinese and English notes about responsibilities, key classes/functions, inputs/outputs, and porting notes.
