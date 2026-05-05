from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = ROOT / "python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import cv2
import ncnn
import numpy as np

from backend_client import BackendClient
from camera_input import CameraInput
from roi_motion import MotionRoiSelector, Roi


DEFAULT_LABELS = ["Correct Wear", "Wrong Wear", "No Wear"]
ASCII_LABELS = DEFAULT_LABELS
TRAFFIC_LIGHT_COLORS = {
    0: (0, 200, 0),
    1: (0, 190, 255),
    2: (0, 0, 255),
}
ROI_COLOR = (255, 255, 0)


@dataclass
class FpsMeter:
    value: float = 0.0
    _last_time: float | None = None

    def update(self) -> float:
        now = time.perf_counter()
        if self._last_time is not None:
            instant = 1.0 / max(now - self._last_time, 1e-6)
            self.value = instant if self.value == 0.0 else self.value * 0.9 + instant * 0.1
        self._last_time = now
        return self.value


@dataclass
class TrackedDetection:
    values: np.ndarray
    last_seen_frame: int

    def is_cached(self, frame_id: int) -> bool:
        return self.last_seen_frame != frame_id


class DetectionCache:
    def __init__(
        self,
        ttl_frames: int = 0,
        match_iou: float = 0.3,
        dedupe_iou: float = 0.45,
    ) -> None:
        self.ttl_frames = max(0, ttl_frames)
        self.match_iou = match_iou
        self.dedupe_iou = dedupe_iou
        self._tracks: list[TrackedDetection] = []

    def update_full(self, detections: np.ndarray, frame_id: int) -> None:
        self._tracks = [
            TrackedDetection(det.astype(np.float32, copy=True), frame_id) for det in detections
        ]
        self._dedupe_tracks()

    def update_motion(
        self,
        detections: np.ndarray,
        frame_id: int,
    ) -> None:
        self.prune(frame_id)
        for det in detections:
            match_idx = self._best_match(det)
            if match_idx is None:
                self._tracks.append(TrackedDetection(det.astype(np.float32, copy=True), frame_id))
            else:
                self._tracks[match_idx] = TrackedDetection(det.astype(np.float32, copy=True), frame_id)
        self._dedupe_tracks()

    def prune(self, frame_id: int) -> None:
        if self.ttl_frames == 0:
            return
        self._tracks = [
            track
            for track in self._tracks
            if frame_id - track.last_seen_frame <= self.ttl_frames
        ]

    def snapshot(self, frame_id: int) -> tuple[np.ndarray, list[bool]]:
        self.prune(frame_id)
        if not self._tracks:
            return np.empty((0, 6), dtype=np.float32), []
        detections = np.stack([track.values for track in self._tracks]).astype(np.float32)
        cached_flags = [track.is_cached(frame_id) for track in self._tracks]
        keep_indices = nms_indices(detections, self.dedupe_iou, mode="class_agnostic")
        detections = detections[keep_indices]
        cached_flags = [cached_flags[idx] for idx in keep_indices]
        return detections, cached_flags

    def active_detections(self, frame_id: int) -> np.ndarray:
        detections, _ = self.snapshot(frame_id)
        return detections

    def _best_match(self, det: np.ndarray) -> int | None:
        best_idx = None
        best_iou = 0.0
        for idx, track in enumerate(self._tracks):
            iou = box_iou(det[:4], track.values[:4])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou < self.match_iou:
            return None
        return best_idx

    def _dedupe_tracks(self) -> None:
        if not self._tracks:
            return
        detections = np.stack([track.values for track in self._tracks]).astype(np.float32)
        keep_indices = nms_indices(detections, self.dedupe_iou, mode="class_agnostic")
        self._tracks = [self._tracks[idx] for idx in keep_indices]


class TextRenderer:
    def __init__(self) -> None:
        self._font = None
        self._pil = None
        try:
            from PIL import Image, ImageDraw, ImageFont

            self._pil = (Image, ImageDraw)
            for font_path in (
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\simhei.ttf",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ):
                if Path(font_path).exists():
                    self._font = ImageFont.truetype(font_path, 18)
                    break
            if self._font is None:
                self._font = ImageFont.load_default()
        except Exception:
            self._pil = None

    def put_text(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.6,
        thickness: int = 1,
    ) -> np.ndarray:
        if self._pil is None:
            cv2.putText(
                image,
                text,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            return image

        Image, ImageDraw = self._pil
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text(origin, text, fill=(color[2], color[1], color[0]), font=self._font)
        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def read_labels(path: Path) -> list[str]:
    if not path.exists():
        return DEFAULT_LABELS.copy()
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return labels or DEFAULT_LABELS.copy()


def label_name(labels: list[str], class_id: int, fallback_ascii: bool = False) -> str:
    source = ASCII_LABELS if fallback_ascii else labels
    if 0 <= class_id < len(source):
        return source[class_id]
    return f"class{class_id}"


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = size - new_w
    pad_h = size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return np.ascontiguousarray(padded), scale, (left, top)


def nms_indices(
    detections: np.ndarray,
    iou_thres: float,
    mode: str = "class_aware",
) -> list[int]:
    if detections.size == 0:
        return []
    if mode not in {"class_aware", "class_agnostic"}:
        raise ValueError(f"Unknown NMS mode: {mode}")

    keep: list[int] = []
    if mode == "class_agnostic":
        groups = [np.arange(detections.shape[0])]
    else:
        groups = [
            np.where(detections[:, 5].astype(np.int32) == class_id)[0]
            for class_id in np.unique(detections[:, 5]).astype(np.int32)
        ]

    for group_indices in groups:
        order = group_indices[detections[group_indices, 4].argsort()[::-1]]

        while order.shape[0] > 0:
            best_idx = int(order[0])
            best = detections[best_idx]
            keep.append(best_idx)
            if order.shape[0] == 1:
                break

            rest_indices = order[1:]
            rest = detections[rest_indices]
            xx1 = np.maximum(best[0], rest[:, 0])
            yy1 = np.maximum(best[1], rest[:, 1])
            xx2 = np.minimum(best[2], rest[:, 2])
            yy2 = np.minimum(best[3], rest[:, 3])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            area_best = max(0.0, best[2] - best[0]) * max(0.0, best[3] - best[1])
            area_rest = np.maximum(0.0, rest[:, 2] - rest[:, 0]) * np.maximum(
                0.0, rest[:, 3] - rest[:, 1]
            )
            union = area_best + area_rest - inter + 1e-6
            order = rest_indices[(inter / union) <= iou_thres]

    return keep


def nms(
    detections: np.ndarray,
    iou_thres: float,
    mode: str = "class_aware",
) -> np.ndarray:
    if detections.size == 0:
        return detections

    keep_indices = nms_indices(detections, iou_thres, mode=mode)

    if not keep_indices:
        return np.empty((0, 6), dtype=np.float32)
    return detections[keep_indices].astype(np.float32)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xx1 = max(float(box_a[0]), float(box_b[0]))
    yy1 = max(float(box_a[1]), float(box_b[1]))
    xx2 = min(float(box_a[2]), float(box_b[2]))
    yy2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def union_roi_with_detections(
    roi: Roi,
    detections: np.ndarray,
    frame_shape: tuple[int, int],
    proximity_pixels: int = 0,
) -> Roi:
    if roi.reason != "motion" or detections.size == 0:
        return roi

    frame_h, frame_w = frame_shape
    gap = max(0, int(proximity_pixels))
    near_x1 = max(0, roi.x1 - gap)
    near_y1 = max(0, roi.y1 - gap)
    near_x2 = min(frame_w, roi.x2 + gap)
    near_y2 = min(frame_h, roi.y2 + gap)
    nearby_mask = (
        (detections[:, 2] >= near_x1)
        & (detections[:, 0] <= near_x2)
        & (detections[:, 3] >= near_y1)
        & (detections[:, 1] <= near_y2)
    )
    nearby = detections[nearby_mask]
    if nearby.size == 0:
        return roi

    x1 = min(float(roi.x1), float(np.min(nearby[:, 0])))
    y1 = min(float(roi.y1), float(np.min(nearby[:, 1])))
    x2 = max(float(roi.x2), float(np.max(nearby[:, 2])))
    y2 = max(float(roi.y2), float(np.max(nearby[:, 3])))
    return Roi(
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(frame_w, int(round(x2))),
        min(frame_h, int(round(y2))),
        roi.reason,
    )


def expand_roi_to_min_size(
    roi: Roi,
    frame_shape: tuple[int, int],
    min_size_pixels: int,
) -> Roi:
    min_size = max(1, int(min_size_pixels))
    if roi.reason == "full" or (roi.width >= min_size and roi.height >= min_size):
        return roi

    frame_h, frame_w = frame_shape
    target_w = min(frame_w, max(roi.width, min_size))
    target_h = min(frame_h, max(roi.height, min_size))
    center_x = (roi.x1 + roi.x2) / 2.0
    center_y = (roi.y1 + roi.y2) / 2.0

    x1 = int(round(center_x - target_w / 2.0))
    y1 = int(round(center_y - target_h / 2.0))
    x1 = min(max(0, x1), max(0, frame_w - target_w))
    y1 = min(max(0, y1), max(0, frame_h - target_h))
    x2 = min(frame_w, x1 + target_w)
    y2 = min(frame_h, y1 + target_h)
    return Roi(x1, y1, x2, y2, roi.reason)


def cached_roi_from_detections(
    detections: np.ndarray,
    frame_shape: tuple[int, int],
    padding_pixels: int,
    min_size_pixels: int = 1,
) -> Roi | None:
    if detections.size == 0:
        return None

    frame_h, frame_w = frame_shape
    pad = max(0, int(padding_pixels))
    x1 = max(0, int(round(float(np.min(detections[:, 0])))) - pad)
    y1 = max(0, int(round(float(np.min(detections[:, 1])))) - pad)
    x2 = min(frame_w, int(round(float(np.max(detections[:, 2])))) + pad)
    y2 = min(frame_h, int(round(float(np.max(detections[:, 3])))) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return expand_roi_to_min_size(Roi(x1, y1, x2, y2, "cached"), frame_shape, min_size_pixels)


def make_input(image: np.ndarray, img_size: int) -> tuple[ncnn.Mat, float, tuple[int, int]]:
    padded, scale, pad = letterbox(image, img_size)
    mat = ncnn.Mat.from_pixels(
        padded,
        ncnn.Mat.PixelType.PIXEL_BGR2RGB,
        img_size,
        img_size,
    )
    mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
    return mat, scale, pad


def decode_output(
    output: ncnn.Mat,
    image_shape: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
    conf_thres: float,
    iou_thres: float,
    nms_mode: str,
) -> np.ndarray:
    pred = output.numpy()
    pred = pred.reshape(-1, pred.shape[-1]).astype(np.float32)
    if pred.shape[1] < 6:
        raise RuntimeError(f"Unexpected output shape: {pred.shape}; need at least 6 values per box")

    boxes = pred[:, :4]
    obj = pred[:, 4:5]
    class_scores = pred[:, 5:]
    scores_by_class = obj * class_scores
    class_ids = scores_by_class.argmax(axis=1)
    scores = scores_by_class[np.arange(scores_by_class.shape[0]), class_ids]

    mask = scores >= conf_thres
    if not np.any(mask):
        return np.empty((0, 6), dtype=np.float32)

    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask].astype(np.float32)

    x_center, y_center, width, height = boxes.T
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    pad_x, pad_y = pad
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    img_h, img_w = image_shape
    x1 = np.clip(x1, 0, img_w - 1)
    y1 = np.clip(y1, 0, img_h - 1)
    x2 = np.clip(x2, 0, img_w - 1)
    y2 = np.clip(y2, 0, img_h - 1)

    detections = np.stack([x1, y1, x2, y2, scores, class_ids], axis=1)
    return nms(detections, iou_thres, mode=nms_mode)


def load_net(args: argparse.Namespace) -> ncnn.Net:
    if not hasattr(ncnn, "Net"):
        raise RuntimeError(
            "The ncnn Python runtime is not available. Install it with "
            "`python -m pip install --target .\\python_packages ncnn` on Windows "
            "or `python3 -m pip install --user ncnn` on Raspberry Pi."
        )

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = args.threads

    ret = net.load_param(str(args.param))
    if ret != 0:
        raise RuntimeError(f"Failed to load param: {args.param} (ret={ret})")
    ret = net.load_model(str(args.bin))
    if ret != 0:
        raise RuntimeError(f"Failed to load bin: {args.bin} (ret={ret})")
    return net


def detect(net: ncnn.Net, image: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    mat, scale, pad = make_input(image, args.img_size)
    extractor = net.create_extractor()
    ret = extractor.input(args.input_name, mat)
    if ret != 0:
        raise RuntimeError(f"Failed to feed input '{args.input_name}' (ret={ret})")

    start = time.perf_counter()
    ret, output = extractor.extract(args.output_name)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if ret != 0:
        raise RuntimeError(f"Failed to extract output '{args.output_name}' (ret={ret})")

    detections = decode_output(
        output,
        image.shape[:2],
        scale,
        pad,
        args.conf_thres,
        args.iou_thres,
        args.nms_mode,
    )
    return detections, elapsed_ms


def detect_roi(
    net: ncnn.Net,
    frame: np.ndarray,
    roi: Roi,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    crop = roi.crop(frame)
    if crop.size == 0:
        return np.empty((0, 6), dtype=np.float32), 0.0

    detections, elapsed_ms = detect(net, crop, args)
    if detections.size:
        detections[:, [0, 2]] += roi.x1
        detections[:, [1, 3]] += roi.y1
        h, w = frame.shape[:2]
        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, w - 1)
        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, h - 1)
    return detections, elapsed_ms


def detection_dicts(
    detections: np.ndarray,
    labels: list[str],
    cached_flags: list[bool] | None = None,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    cached_flags = cached_flags or [False] * len(detections)
    for idx, (x1, y1, x2, y2, score, class_id) in enumerate(detections):
        cid = int(class_id)
        payload.append(
            {
                "class_id": cid,
                "label": label_name(labels, cid),
                "confidence": round(float(score), 4),
                "cached": bool(cached_flags[idx]),
                "box": {
                    "x1": int(round(float(x1))),
                    "y1": int(round(float(y1))),
                    "x2": int(round(float(x2))),
                    "y2": int(round(float(y2))),
                },
            }
        )
    return payload


def build_backend_payload(
    args: argparse.Namespace,
    frame: np.ndarray,
    frame_id: int,
    roi: Roi,
    detections: np.ndarray,
    labels: list[str],
    fps: float,
    inference_ms: float,
    cached_flags: list[bool] | None = None,
) -> dict[str, Any]:
    h, w = frame.shape[:2]
    return {
        "device_id": args.device_id,
        "timestamp_ms": int(time.time() * 1000),
        "frame_id": frame_id,
        "image_size": {"width": w, "height": h},
        "roi": roi.to_dict(),
        "fps": round(float(fps), 2),
        "inference_ms": round(float(inference_ms), 2),
        "detections": detection_dicts(detections, labels, cached_flags),
    }


def draw_label(
    image: np.ndarray,
    text_renderer: TextRenderer,
    text: str,
    p1: tuple[int, int],
    color: tuple[int, int, int],
) -> np.ndarray:
    display_text = text
    (tw, th), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    label_y = max(p1[1], th + baseline + 6)
    cv2.rectangle(
        image,
        (p1[0], label_y - th - baseline - 6),
        (p1[0] + tw + 10, label_y),
        color,
        -1,
    )
    return text_renderer.put_text(image, display_text, (p1[0] + 5, label_y - baseline - 4), (0, 0, 0))


def draw_detections(
    image: np.ndarray,
    detections: np.ndarray,
    labels: list[str],
    text_renderer: TextRenderer,
) -> np.ndarray:
    out = image.copy()
    for x1, y1, x2, y2, score, class_id in detections:
        cid = int(class_id)
        color = TRAFFIC_LIGHT_COLORS.get(cid, (255, 255, 255))
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, p1, p2, color, 2)
        text = f"{label_name(labels, cid)} {score:.2f}"
        out = draw_label(out, text_renderer, text, p1, color)
    return out


def draw_status(
    image: np.ndarray,
    text_renderer: TextRenderer,
    fps: float,
    inference_ms: float,
    roi: Roi | None,
    detections: np.ndarray,
    skipped: bool,
) -> np.ndarray:
    status_text = "Status: Sleep (No Motion)" if skipped else "Status: Detecting"
    status_color = (150, 150, 150) if skipped else (0, 0, 255)
    lines = [
        status_text,
        f"FPS: {fps:.1f}",
        f"Infer: {inference_ms:.1f} ms" if not skipped else "Infer: skipped",
        f"Detections: {len(detections)}",
    ]
    if skipped:
        lines.append("Using last detections")
    elif roi:
        lines.append(f"ROI: {roi.reason} [{roi.x1},{roi.y1},{roi.x2},{roi.y2}]")
        cv2.rectangle(image, (roi.x1, roi.y1), (roi.x2, roi.y2), ROI_COLOR, 2)
        image = draw_label(image, text_renderer, f"ROI: {roi.reason}", (roi.x1, roi.y1), ROI_COLOR)
    else:
        lines.append("ROI: none")

    panel_height = 22 * len(lines) + 18
    cv2.rectangle(image, (8, 8), (430, 8 + panel_height), (0, 0, 0), -1)
    y = 32
    for idx, line in enumerate(lines):
        color = status_color if idx == 0 else (255, 255, 255)
        image = text_renderer.put_text(image, line, (16, y), color, scale=0.6, thickness=1)
        y += 22
    return image


def run_image(args: argparse.Namespace, net: ncnn.Net, labels: list[str]) -> None:
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    full_roi = Roi(0, 0, image.shape[1], image.shape[0], "full")
    detections, elapsed_ms = detect_roi(net, image, full_roi, args)
    output_path = args.output
    if output_path is None:
        output_path = args.image.with_name(f"{args.image.stem}_ncnn{args.image.suffix}")

    rendered = draw_detections(image, detections, labels, TextRenderer())
    rendered = draw_status(rendered, TextRenderer(), 0.0, elapsed_ms, full_roi, detections, skipped=False)
    if not cv2.imwrite(str(output_path), rendered):
        raise RuntimeError(f"Could not write output image: {output_path}")

    print(f"image: {args.image}")
    print(f"output: {output_path}")
    print(f"detections: {len(detections)}")
    print(f"inference: {elapsed_ms:.2f} ms")


def run_camera(args: argparse.Namespace, net: ncnn.Net, labels: list[str]) -> None:
    camera = CameraInput(
        camera_index=args.camera_index,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    if not camera.is_opened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    roi_selector = MotionRoiSelector(
        mode=args.roi_mode,
        full_frame_interval=args.full_frame_interval,
        min_area=args.roi_min_area,
        padding=args.roi_padding,
        padding_pixels=args.roi_padding_pixels,
        history=args.mog2_history,
        var_threshold=args.mog2_var_threshold,
        hold_frames=args.roi_hold_frames,
        smooth_alpha=args.roi_smooth_alpha,
    )
    backend = BackendClient(args.backend_url, timeout=args.backend_timeout, max_queue=args.backend_queue)
    text_renderer = TextRenderer()
    fps_meter = FpsMeter()
    detection_cache = DetectionCache(ttl_frames=args.detection_ttl_frames, dedupe_iou=args.iou_thres)

    frame_id = 0
    last_inference_ms = 0.0
    last_face_refresh_time = 0.0
    last_full_refresh_time = 0.0

    if not args.headless:
        print("Press q or Esc to quit.")

    try:
        while True:
            ok, frame = camera.read_frame()
            if not ok or frame is None:
                break

            frame_id += 1
            now = time.perf_counter()
            fps = fps_meter.update()
            roi = roi_selector.select(frame, frame_id)
            active_detections = detection_cache.active_detections(frame_id)
            full_refresh_due = (
                args.full_frame_refresh_ms > 0
                and last_full_refresh_time > 0.0
                and (now - last_full_refresh_time) * 1000.0 >= args.full_frame_refresh_ms
            )
            if full_refresh_due and args.roi_mode == "hybrid":
                h, w = frame.shape[:2]
                roi = Roi(0, 0, w, h, "full")

            if roi is not None:
                roi = union_roi_with_detections(
                    roi,
                    active_detections,
                    frame.shape[:2],
                    proximity_pixels=args.roi_padding_pixels,
                )
            elif (
                active_detections.size > 0
                and (
                    (
                        args.face_refresh_ms > 0
                        and (now - last_face_refresh_time) * 1000.0 >= args.face_refresh_ms
                    )
                    or (
                        args.cached_roi_interval > 0
                        and frame_id % args.cached_roi_interval == 0
                    )
                )
            ):
                roi = cached_roi_from_detections(
                    active_detections,
                    frame.shape[:2],
                    args.cached_roi_padding_pixels,
                    args.roi_min_size_pixels,
                )
            if roi is not None:
                roi = expand_roi_to_min_size(roi, frame.shape[:2], args.roi_min_size_pixels)
            skipped = roi is None

            if roi is not None:
                raw_detections, inference_ms = detect_roi(net, frame, roi, args)
                if roi.reason == "full":
                    detection_cache.update_full(raw_detections, frame_id)
                    last_full_refresh_time = now
                else:
                    detection_cache.update_motion(raw_detections, frame_id)
                last_face_refresh_time = now
                detections, cached_flags = detection_cache.snapshot(frame_id)
                last_inference_ms = inference_ms
                payload = build_backend_payload(
                    args,
                    frame,
                    frame_id,
                    roi,
                    detections,
                    labels,
                    fps,
                    inference_ms,
                    cached_flags,
                )
                backend.submit(payload)
            else:
                detections, cached_flags = detection_cache.snapshot(frame_id)
                inference_ms = last_inference_ms

            if args.headless:
                continue

            rendered = draw_detections(frame, detections, labels, text_renderer)
            rendered = draw_status(
                rendered,
                text_renderer,
                fps,
                inference_ms,
                roi,
                detections,
                skipped=skipped,
            )
            cv2.imshow("ncnn yolov5 camera", rendered)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        backend.close()
        camera.release()
        if not args.headless:
            cv2.destroyAllWindows()


def run_self_test(args: argparse.Namespace, net: ncnn.Net) -> None:
    image = np.full((args.img_size, args.img_size, 3), 114, dtype=np.uint8)
    full_roi = Roi(0, 0, image.shape[1], image.shape[0], "full")
    detections, elapsed_ms = detect_roi(net, image, full_roi, args)
    print("environment: ok")
    print(f"model: {args.param.name} + {args.bin.name}")
    print(f"input: {args.input_name}, output: {args.output_name}, size: {args.img_size}")
    print(f"labels: {', '.join(DEFAULT_LABELS)}")
    print(f"blank-image detections: {len(detections)}")
    print(f"inference: {elapsed_ms:.2f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv5 NCNN camera inference.")
    parser.add_argument("--param", type=Path, default=ROOT / "best.ncnn.param")
    parser.add_argument("--bin", type=Path, default=ROOT / "best.ncnn.bin")
    parser.add_argument("--labels", type=Path, default=ROOT / "labels.txt")
    parser.add_argument("--image", type=Path, help="Image path for single-image inference.")
    parser.add_argument("--output", type=Path, help="Output image path.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera", type=int, dest="camera_index", help=argparse.SUPPRESS)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=640)
    parser.add_argument("--camera-fps", type=int)
    parser.add_argument("--headless", action="store_true", help="Do not open an OpenCV preview window.")
    parser.add_argument("--self-test", action="store_true", help="Run one inference on a blank image.")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--input-name", default="in0")
    parser.add_argument("--output-name", default="out0")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--nms-mode", choices=("class_aware", "class_agnostic"), default="class_agnostic")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--roi-mode", choices=("hybrid", "roi", "full"), default="hybrid")
    parser.add_argument("--full-frame-interval", type=int, default=0)
    parser.add_argument("--full-frame-refresh-ms", type=float, default=2000.0)
    parser.add_argument("--roi-min-area", type=int, default=800)
    parser.add_argument("--roi-min-size-pixels", type=int, default=96)
    parser.add_argument("--roi-padding", type=float, default=0.15)
    parser.add_argument("--roi-padding-pixels", type=int, default=50)
    parser.add_argument("--roi-hold-frames", type=int, default=0)
    parser.add_argument("--roi-smooth-alpha", type=float, default=0.6)
    parser.add_argument("--face-refresh-ms", type=float, default=500.0)
    parser.add_argument("--cached-roi-interval", type=int, default=0)
    parser.add_argument("--cached-roi-padding-pixels", type=int, default=60)
    parser.add_argument("--detection-ttl-frames", type=int, default=0)
    parser.add_argument("--mog2-history", type=int, default=500)
    parser.add_argument("--mog2-var-threshold", type=float, default=25.0)
    parser.add_argument("--backend-url")
    parser.add_argument("--backend-timeout", type=float, default=1.0)
    parser.add_argument("--backend-queue", type=int, default=8)
    parser.add_argument("--device-id", default=platform.node() or "camera-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_labels(args.labels)
    net = load_net(args)

    if args.self_test:
        run_self_test(args, net)
    elif args.image is not None:
        run_image(args, net, labels)
    else:
        run_camera(args, net, labels)


if __name__ == "__main__":
    main()
