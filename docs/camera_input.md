# camera_input.py

## 中文说明

### 模块职责

`camera_input.py` 封装摄像头读取逻辑，让主检测程序只关心“能不能读到一帧 BGR 图像”。后续移植到树莓派时，优先修改这个文件，而不是改主检测流程。

### 主要类

- `CameraInput`：摄像头输入源。

### 主要方法

- `__post_init__()`：创建 OpenCV `VideoCapture`。Windows 使用 `cv2.CAP_DSHOW`，Linux/树莓派使用默认后端；同时设置宽、高和缓冲区大小。
- `is_opened()`：返回摄像头是否成功打开。
- `read_frame()`：读取一帧图像。返回 `(True, frame)` 或 `(False, None)`；图像会保证是 `uint8`、BGR、三通道。
- `release()`：释放摄像头资源。

### 树莓派替换点

如果使用 USB 摄像头，通常不用改代码，继续使用 `cv2.VideoCapture(0)`。如果使用树莓派 CSI 摄像头，可以在本文件中改为 Picamera2，并在返回前把 RGB 转成 BGR。主程序要求保持接口不变：

```python
def read_frame(self) -> tuple[bool, np.ndarray | None]:
    ...
```

### 输入输出

- 输入：摄像头编号、宽度、高度。
- 输出：BGR 格式的 OpenCV 图像帧。

## English Notes

### Module Purpose

`camera_input.py` wraps camera capture so the detector loop only needs a simple source that returns BGR frames. On Raspberry Pi, this is the preferred file to change when the camera source changes.

### Key Class

- `CameraInput`: Camera frame source.

### Key Methods

- `__post_init__()`: Opens OpenCV `VideoCapture`; uses `CAP_DSHOW` on Windows and the default backend elsewhere.
- `is_opened()`: Reports whether the camera opened successfully.
- `read_frame()`: Returns `(True, frame)` or `(False, None)` and normalizes frames to `uint8` BGR with three channels.
- `release()`: Releases the camera handle.

### Raspberry Pi Porting Point

USB cameras usually work with `cv2.VideoCapture(0)`. For the Raspberry Pi camera module, replace the internals with Picamera2 and convert RGB frames to BGR before returning. Keep the public `read_frame()` contract unchanged.
