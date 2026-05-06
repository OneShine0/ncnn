# detect_ncnn_yolov5.py

## 中文说明

### 模块职责

`detect_ncnn_yolov5.py` 是项目主入口，负责加载 YOLOv5 NCNN 模型，读取图片、视频或摄像头画面，完成预处理、推理、后处理、ROI 优化、画框显示和后端上报。

### 主要类

- `FpsMeter`：统计循环 FPS，并用平滑方式减少 FPS 数字抖动。
- `TrackedDetection`：保存单个检测框和最后出现的帧号，用于判断检测框是否来自缓存。
- `DetectionCache`：维护检测框缓存。全帧检测会替换缓存，ROI 检测会更新或追加缓存，缓存结果再经过 class-agnostic NMS 去重。
- `TextRenderer`：优先用 PIL 字体绘制中英文文本，失败时退回 OpenCV 文本绘制。

### 主要函数

- `read_labels(path)`：读取 `labels.txt`，如果文件不存在或为空，则使用默认类别。
- `label_name(labels, class_id, fallback_ascii=False)`：根据类别编号返回显示名称。
- `letterbox(image, size)`：按比例缩放图像并填充到模型输入尺寸。
- `make_input(image, img_size)`：把 BGR 图像转换成 NCNN `Mat`，完成 BGR 到 RGB、归一化和 letterbox。
- `decode_output(...)`：解析模型输出，计算 `obj * class_score`，还原坐标并执行 NMS。
- `nms_indices(...)` / `nms(...)`：按类别或跨类别去重重叠检测框。
- `box_iou(box_a, box_b)`：计算两个框的 IoU。
- `union_roi_with_detections(...)`：把运动 ROI 与附近缓存人脸框合并，避免 ROI 裁掉目标。
- `expand_roi_to_min_size(...)`：把过小 ROI 扩展到最低尺寸。
- `cached_roi_from_detections(...)`：根据已有缓存检测框生成复查 ROI。
- `load_net(args)`：加载 `.param` 和 `.bin` 模型文件，设置线程数。
- `detect(net, image, args)`：对单张图像执行一次完整推理。
- `detect_roi(net, frame, roi, args)`：裁剪 ROI 后推理，并把检测框坐标映射回原图。
- `detection_dicts(...)`：把检测结果转换成可 JSON 序列化的列表。
- `build_backend_payload(...)`：构造发送给后端的 JSON 数据。
- `draw_label(...)`、`draw_detections(...)`、`draw_status(...)`：绘制标签、检测框和状态面板。
- `run_image(...)`：图片检测入口。
- `run_video(...)`：视频文件检测入口，支持预览、`--headless`、可选 `--output` 保存结果视频。
- `run_camera(...)`：摄像头实时检测入口。
- `run_self_test(...)`：用空白图像验证模型和环境是否能完成一次推理。
- `parse_args()`：定义命令行参数。
- `main()`：根据 `--self-test`、`--image`、`--video` 选择运行模式，默认运行摄像头模式。

### 输入输出

- 输入模型：`best.ncnn.param`、`best.ncnn.bin`。
- 输入标签：`labels.txt`。
- 图片输入：`--image path`。
- 视频输入：`--video path`。
- 摄像头输入：`--camera-index 0`。
- 输出图片或视频：`--output path`。
- 后端输出：`--backend-url http://HOST:PORT/api/detections`。

### 调用关系

`main()` 先读取参数和标签，再加载 NCNN 模型。图片模式直接执行一次 `detect_roi()`。视频和摄像头模式循环读取帧，先用 `MotionRoiSelector` 选择 ROI，再执行 `detect_roi()`，更新 `DetectionCache`，最后绘制画面并可选发送后端 JSON。

## English Notes

### Module Purpose

`detect_ncnn_yolov5.py` is the main entry point. It loads the YOLOv5 NCNN model, reads images, videos, or camera frames, then performs preprocessing, inference, postprocessing, ROI optimization, drawing, preview, and optional backend reporting.

### Key Classes

- `FpsMeter`: Tracks loop FPS with smoothing.
- `TrackedDetection`: Stores one detection and the last frame where it was refreshed.
- `DetectionCache`: Keeps stable detections across ROI frames, treats non-full ROI results as regional evidence, expires stale boxes with TTL, and reports TTL-expired boxes for local visualization.
- `RefreshTileScheduler`: Time-slices periodic refreshes across the left half, right half, and one centered vertical patch.
- `TextRenderer`: Draws labels with PIL fonts when available, with OpenCV text as fallback.

### Key Functions

- `read_labels(path)`: Loads labels from disk with defaults as fallback.
- `letterbox(image, size)`: Resizes and pads an image to the model input size.
- `make_input(image, img_size)`: Converts BGR frames to normalized NCNN input.
- `decode_output(...)`: Converts raw model rows into boxes, scores, class ids, and applies NMS.
- `refresh_tiles(...)`: Builds the 5-tile refresh layout from the current frame shape.
- `load_net(args)`: Loads NCNN model files and configures thread count.
- `detect(...)`: Runs one full inference on an image.
- `detect_roi(...)`: Runs inference on a crop and maps boxes back to full-frame coordinates.
- `run_image(...)`: Handles still-image inference.
- `run_video(...)`: Handles video-file inference with optional preview, headless mode, output video, and backend reporting.
- `run_camera(...)`: Handles live camera inference.
- `run_self_test(...)`: Checks whether the runtime and model can run one blank-frame inference.
- `main()`: Selects self-test, image, video, or camera mode.

### Inputs and Outputs

- Model inputs: `best.ncnn.param`, `best.ncnn.bin`.
- Label input: `labels.txt`.
- Image input: `--image path`.
- Video input: `--video path`.
- Camera input: `--camera-index 0`.
- Rendered output: `--output path`.
- Backend JSON output: `--backend-url http://HOST:PORT/api/detections`.

### Flow

`main()` parses arguments, loads labels, and loads the NCNN network. Image mode runs a single full-frame inference. Video and camera modes read frames in a loop, prefer motion/cached ROIs, then use time-sliced refresh tiles when idle. Each ROI runs through `detect_roi()`, updates `DetectionCache`, renders results, and optionally submits JSON payloads to the backend.
