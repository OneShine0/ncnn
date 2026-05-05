# Raspberry Pi deployment and backend handoff

## Camera input contract

`camera_input.py` is the only file that should need camera-source changes on
the Pi. Keep this public interface:

```python
class CameraInput:
    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        ...

    def release(self) -> None:
        ...
```

`read_frame()` must return:

- `ok=True` and a frame when capture succeeds.
- `ok=False, None` when the stream ends or fails.
- Frame format: BGR, `uint8`, shape `height x width x 3`.

Possible Raspberry Pi implementations:

- USB camera: keep OpenCV `cv2.VideoCapture(0)`.
- Raspberry Pi camera module: use Picamera2, then convert RGB to BGR.
- Monitoring stream: use `cv2.VideoCapture(rtsp_or_http_url)`.

The rest of the detector does not care where the frame came from.

## Backend HTTP contract

Run with:

```bash
python3 detect_ncnn_yolov5.py \
  --headless \
  --backend-url http://HOST:PORT/api/detections \
  --device-id pi-camera-01
```

The detector sends an HTTP `POST` with JSON. Example:

```json
{
  "device_id": "pi-camera-01",
  "timestamp_ms": 1777890000000,
  "frame_id": 128,
  "image_size": {"width": 640, "height": 640},
  "roi": {
    "x1": 112,
    "y1": 80,
    "x2": 420,
    "y2": 380,
    "width": 308,
    "height": 300,
    "reason": "motion"
  },
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

Notes for backend developers:

- Coordinates are in the original full-frame coordinate system, not ROI-local coordinates.
- `roi.reason` is usually `motion`, `cached`, or `full`.
- Overlapping mask-status classes are suppressed with class-agnostic NMS, so one face should produce one detection with the highest-confidence label.
- `cached=true` means the box comes from the local detection cache and was not freshly detected on the current frame.
- The client sends only after a model inference runs. Frames skipped because no motion was detected are not posted.
- A skipped preview frame means the model did not run on the current frame; Windows preview keeps the last detection boxes and displays `Using last detections`.
- HTTP send happens on a background thread. If the backend is slow, old payloads are dropped in favor of newer ones.
- Return any `2xx` response; the current client does not require a response body.

## ROI and speed strategy

The Pi default should be:

```bash
--nms-mode class_agnostic --roi-mode hybrid --full-frame-interval 0 --full-frame-refresh-ms 2000 --roi-min-area 800 --roi-min-size-pixels 96 --roi-padding-pixels 50 --roi-hold-frames 0 --roi-smooth-alpha 0.6 --face-refresh-ms 500 --cached-roi-interval 0 --cached-roi-padding-pixels 60 --mog2-history 500 --mog2-var-threshold 25.0 --detection-ttl-frames 0
```

How it works:

- Class-agnostic NMS and class-agnostic cache matching keep one box per face even when the mask-status class flickers.
- OpenCV `cv2.createBackgroundSubtractorMOG2()` finds motion with `history=500`, `varThreshold=25`, and `detectShadows=False`.
- The motion mask is denoised with OpenCV thresholding and ellipse-kernel open/close morphology.
- OpenCV contours and contour-area filtering remove small motion noise and merge valid motion regions into one ROI.
- The ROI uses fixed pixel padding by default (`--roi-padding-pixels 50`) and a minimum crop size (`--roi-min-size-pixels 96`), then is resized/letterboxed to `320x320` and sent to NCNN.
- Frame 1 and every `--full-frame-refresh-ms 2000` milliseconds use full-frame detection; set `--full-frame-interval` above `0` only if you want old frame-count scheduling too.
- Motion ROI is unioned only with nearby cached detections so unrelated movement does not expand the ROI across the whole frame.
- If MOG2 misses subtle motion but cached detections exist, `--face-refresh-ms 500` runs a padded cached-face ROI recheck. The older frame-count cached ROI fallback stays disabled by default with `--cached-roi-interval 0`.
- `--detection-ttl-frames 0` means cached detections do not expire by time.
- Full-frame detection replaces the whole cache, so it is responsible for clearing ghost boxes.

Important limitation:

- A ghost box can remain until the next scheduled full-frame detection. Lower `--full-frame-refresh-ms` to clear it faster, or raise it for more speed. Raise `--face-refresh-ms` if cached ROI rechecks cost too much CPU.

## Raspberry Pi production advice

- Use `--headless`; OpenCV windows cost CPU and may fail without a desktop session.
- Capture at `640x640` first. If the camera cannot provide that exact size, OpenCV uses the closest supported size and coordinates follow the actual frame.
- Use `--threads 4` on a 4-core Pi, then benchmark `2` and `3` if the camera pipeline stutters.
- Tune `--roi-padding-pixels` first for coverage. Try `40` to `60` when a moving person is clipped at the ROI edge.
- Tune `--face-refresh-ms` next: lower values are steadier for slight head turns, higher values save CPU.
- Start with CPU NCNN. Try Vulkan only after the CPU path is stable and your Pi image has working drivers.
- If FPS is still not enough, the next best upgrades are INT8 quantization and a C++ NCNN implementation with the same JSON contract.

## C++ fallback path

Install tools:

```bash
sudo apt update
sudo apt install -y build-essential git cmake libopencv-dev
```

Build NCNN:

```bash
git clone --depth=1 https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
mkdir -p build
cd build
cmake -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON ..
cmake --build . -j$(nproc)
sudo cmake --install .
```

Keep the same model contract:

- Input blob: `in0`
- Output blob: `out0`
- Input size: `320x320`
- Recommended camera capture size: `640x640`
- Preprocess: BGR to RGB, letterbox with value `114`, normalize by `1/255`
- Output row format: `x, y, w, h, obj, class0, class1, class2`
- Final score: `obj * class_score`
- NMS: per class
