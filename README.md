# YOLOv5 NCNN camera detector

This folder runs your exported YOLOv5 NCNN model on a camera stream.

Model files:

- `best.ncnn.param`
- `best.ncnn.bin`

Class mapping:

- `0`: Correct Wear, green box
- `1`: Wrong Wear, yellow/orange box
- `2`: No Wear, red box

## Windows usage

Default camera preview:

```powershell
cd C:\Users\71549\Desktop\ncnn
python .\detect_ncnn_yolov5.py
```

The preview shows detection boxes, FPS, inference time, and the current ROI.
When a frame has no valid motion ROI, inference is skipped and the preview says
`Using last detections`; in that state no stale ROI box is drawn.
The camera defaults to `640x640`, while each full-frame or ROI crop is still
letterboxed to the model input size `320x320`.
Press `q` or `Esc` to quit.

Environment/model self-test:

```powershell
python .\detect_ncnn_yolov5.py --self-test
```

Single image test:

```powershell
python .\detect_ncnn_yolov5.py --image .\test.jpg --output .\result.jpg
```

Headless mode with backend reporting:

```powershell
python .\detect_ncnn_yolov5.py --headless --backend-url http://HOST:PORT/api/detections --device-id win-camera-01
```

## Important options

```text
--camera-index 0
--camera-width 640
--camera-height 640
--camera-fps 30
--img-size 320
--conf-thres 0.25
--iou-thres 0.45
--nms-mode class_agnostic
--threads 4
--roi-mode hybrid
--full-frame-interval 0
--full-frame-refresh-ms 2000
--roi-min-area 800
--roi-min-size-pixels 96
--roi-padding 0.15
--roi-padding-pixels 50
--roi-hold-frames 0
--roi-smooth-alpha 0.6
--face-refresh-ms 500
--cached-roi-interval 0
--cached-roi-padding-pixels 60
--mog2-history 500
--mog2-var-threshold 25.0
--detection-ttl-frames 0
--backend-url http://HOST:PORT/api/detections
--device-id pi-camera-01
```

ROI modes:

- `hybrid`: default. Run full-frame detection on frame 1, on the time refresh, or on `--full-frame-interval` if it is greater than `0`; otherwise use OpenCV MOG2 motion ROI.
- `roi`: run full-frame detection on frame 1, then only run OpenCV MOG2 motion ROI.
- `full`: run full-frame detection every frame. Most stable, slowest.

`--nms-mode class_agnostic` is the default for mask-wearing detection. It keeps
one highest-confidence result when the same face overlaps across `Correct Wear`,
`Wrong Wear`, and `No Wear`, which prevents one face from showing multiple
boxes because its class score flickers.

The motion ROI is based on `cv2.createBackgroundSubtractorMOG2()`. Defaults are
aligned with the Raspberry Pi reference script: `history=500`,
`varThreshold=25`, and `detectShadows=False`. OpenCV thresholding,
ellipse-kernel open/close morphology, and contour area filtering create one
merged ROI for inference. The lower threshold and smaller default contour area
make small face turns easier to catch.

Motion ROI detections update a small detection cache instead of replacing the
whole frame result. This prevents a hand-only ROI from clearing a stable face
box. Cache matching is class-agnostic, so a face that changes from `Correct
Wear` to `Wrong Wear` updates the same track instead of creating a second box.
If MOG2 reports no motion but cached detections exist, the detector runs a
padded cached-face ROI recheck about every `--face-refresh-ms` milliseconds.
The older `--cached-roi-interval` frame-based fallback is still available for
compatibility and stays disabled by default with `--cached-roi-interval 0`. Ghost boxes
are cleared by the periodic full-frame refresh, because a full-frame inference
replaces the whole detection cache. `--detection-ttl-frames 0` means there is no
time-based expiry.

## Raspberry Pi quick start

Copy these files to the Pi:

```text
best.ncnn.param
best.ncnn.bin
detect_ncnn_yolov5.py
camera_input.py
roi_motion.py
backend_client.py
labels.txt
requirements.txt
RASPBERRY_PI_DEPLOYMENT.md
```

Install dependencies:

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv
python3 -m pip install --user ncnn numpy
```

Run without display and post detections:

```bash
python3 detect_ncnn_yolov5.py \
  --headless \
  --backend-url http://HOST:PORT/api/detections \
  --device-id pi-camera-01 \
  --camera-width 640 \
  --camera-height 640 \
  --threads 4
```

If `pip install ncnn` has no wheel for your Pi OS/Python version, use the C++
production route from the deployment document.

## Performance notes

- Keep `--headless` enabled on the Pi.
- Capture at a modest square resolution such as `640x640`; the model input remains `320x320`.
- Start with the default `--roi-mode hybrid`; lower `--full-frame-refresh-ms` to clear ghost boxes faster, or raise it for more speed.
- Raise `--roi-min-area` if small lighting changes trigger too many ROIs.
- Tune `--roi-padding-pixels` first when adjusting ROI coverage; use `40` to `60` if fast movement clips the person.
- Raise `--face-refresh-ms` for more speed, or lower it if a still face updates too slowly.
- Keep `--full-frame-interval 0` unless you specifically want old FPS-dependent full-frame scheduling.
- If Python is still too slow, migrate the same preprocessing, ROI, and JSON contract to C++ NCNN.
