# raw_stream.py

## 中文说明

`raw_stream.py` 提供检测端自带的原始 MJPEG 视频流备用方案。当前推荐优先使用树莓派 `mjpg-streamer` 输出原始视频流；只有没有外部推流程序时，才让检测端自己编码原始 BGR 帧。它不绘制检测框、ROI、TTL 或状态栏。PC 后端拉取这个流，再用 JSON 检测结果自行渲染覆盖层。

### 主要类

- `RawFrameHub`：保存最新 JPEG 帧，支持 FPS 限流、按宽度等比缩放和 JPEG quality。
- `RawMjpegServer`：基于 Python 标准库 HTTP server 提供 `GET /stream.mjpg`。

### 端点

```text
GET /stream.mjpg
Content-Type: multipart/x-mixed-replace; boundary=frame
```

### 注意事项

- 默认不启动，需显式传 `--raw-stream`。
- 当前推荐树莓派运行 `mjpg-streamer`，PC 检测端用 `--stream-url` 读取它；`--raw-stream` 只作为备用。
- 启用备用方案时，可监听 `0.0.0.0:8090`，PC 后端通过局域网访问。
- 视频流不包含检测结果，检测结果仍通过 `--backend-url` JSON 上报。

## English Notes

`raw_stream.py` is a fallback raw MJPEG server inside the detector process. The recommended path is Raspberry Pi `mjpg-streamer` plus PC-side detection with `--stream-url`. This fallback does not draw boxes or status overlays, and the PC backend should combine the stream with the latest JSON detection payload.
