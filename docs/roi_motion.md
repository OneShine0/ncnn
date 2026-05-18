# roi_motion.py

## 中文说明

### 模块职责

`roi_motion.py` 根据画面运动区域生成 ROI，帮助主程序只在需要的位置运行模型，从而提升实时检测速度。它不执行模型推理，只负责给出“下一帧应该检测哪里”。

### 主要类

- `Roi`：不可变数据类，表示一个矩形区域。
- `MotionRoiSelector`：根据首帧全图策略和 OpenCV MOG2/帧差运动检测选择 ROI。

### `Roi` 方法

- `width` / `height`：计算 ROI 宽高。
- `crop(frame)`：从原图裁剪 ROI。
- `to_dict()`：转换成后端 JSON 中使用的字典。

### `MotionRoiSelector` 方法

- `__init__(...)`：设置模式、最小运动面积、padding、MOG2/帧差参数、ROI 保持帧数和平滑系数。
- `select(frame, frame_id)`：主入口。根据当前帧号和模式返回全帧 ROI、运动 ROI、短暂保持的旧 ROI，或者 `None`。
- `_motion_roi(frame)`：根据 `--motion-source` 使用 MOG2、帧差或二者并集生成运动区域。
- `_stabilize_motion_roi(roi, frame_shape)`：对连续 ROI 做平滑，减少检测区域抖动。

### ROI 模式

- `full`：每帧全图检测，最稳定但最慢。
- `roi`：第一帧全图，之后主要依赖运动区域；缓存 ROI 由主程序的 `--cached-roi-ms` 控制。
- `hybrid`：默认模式。当前默认关闭缓存 ROI 和周期刷新，因此首帧全图后主要依赖运动 ROI；设置正数间隔后才启用缓存 ROI 或周期刷新 ROI。

### 调参建议

- 误触发太多：增大 `--roi-min-area` 或 `--mog2-var-threshold`。
- 人脸被裁掉：增大 `--roi-padding-pixels`。
- 静止缓存框需要复查：设置 `--cached-roi-ms` 为正数。
- 需要周期兜底刷新：设置 `--full-frame-refresh-ms` 为正数。

## English Notes

### Module Purpose

`roi_motion.py` finds a region of interest from frame motion so the detector can run on smaller crops. It does not run inference; it only decides where inference should happen next.

### Key Classes

- `Roi`: Immutable rectangle data object.
- `MotionRoiSelector`: Selects full-frame or motion-based ROIs using OpenCV MOG2 by default, with frame diff still available by command-line switch.

### Key Methods

- `Roi.crop(frame)`: Crops the ROI from a frame.
- `Roi.to_dict()`: Converts ROI data to the backend JSON shape.
- `MotionRoiSelector.select(frame, frame_id)`: Returns a full-frame ROI, motion ROI, held ROI, or `None`.
- `_motion_roi(frame)`: Builds motion boxes from `--motion-source frame_diff`, `mog2`, or `both`, then applies padding and ROI validation.
- `_stabilize_motion_roi(...)`: Smooths ROI movement across frames.

### Tuning

Increase `--roi-padding-pixels` if targets are clipped, set a positive `--cached-roi-ms` when cached boxes need timed rechecks, and set a positive `--detection-ttl-ms` only when you want time-based expiry diagnostics. Non-full ROI results are regional positive and negative evidence: old boxes centered inside a checked ROI are cleared if the model does not detect them again. The recommended default is `--motion-source mog2`; use `--motion-source frame_diff` for frame-diff-only testing, or `--motion-source both` to merge MOG2 and frame diff. The default `--mog2-history 80` and `--mog2-var-threshold 25.0` are retained. The default `--roi-smooth-alpha 0.6` keeps the older stable ROI feel: higher values are steadier but trail movement more, while lower values react faster but can jitter. For frame diff, lower `--frame-diff-threshold` is more sensitive but noisier, lower `--frame-diff-min-area` catches smaller changes but can false-trigger, and `--frame-diff-alpha` controls how quickly the reference image adapts. Periodic refresh tiles are disabled by default with `--full-frame-refresh-ms 0`; set a positive value to check the left half, right half, and one centered vertical patch as an idle fallback. Discuss the expected effect before changing MOG2/ROI defaults.
