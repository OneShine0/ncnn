from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import socket
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
DEFAULT_STREAM_URL = "http://127.0.0.1:8080/?action=stream"
DEFAULT_BACKEND_URL = "http://172.20.10.3:5000/api/detections"
DEFAULT_LATEST_JSON_HOST = "0.0.0.0"
DEFAULT_LATEST_JSON_PORT = 8090


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
        self._expired_tracks: list[TrackedDetection] = []

    def set_ttl_frames(self, ttl_frames: int) -> None:
        self.ttl_frames = max(0, int(ttl_frames))

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

    def update_region(
        self,
        detections: np.ndarray,
        frame_id: int,
        roi: Roi,
    ) -> None:
        self.prune(frame_id)
        self._tracks = [
            track for track in self._tracks if not box_center_in_roi(track.values[:4], roi)
        ]
        for det in detections:
            self._tracks.append(TrackedDetection(det.astype(np.float32, copy=True), frame_id))
        self._dedupe_tracks()

    def prune(self, frame_id: int) -> list[TrackedDetection]:
        if self.ttl_frames == 0:
            return []
        kept: list[TrackedDetection] = []
        expired: list[TrackedDetection] = []
        for track in self._tracks:
            if frame_id - track.last_seen_frame <= self.ttl_frames:
                kept.append(track)
            else:
                expired.append(track)
        self._tracks = kept
        self._expired_tracks.extend(expired)
        return expired

    def pop_expired_tracks(self) -> list[TrackedDetection]:
        expired = self._expired_tracks
        self._expired_tracks = []
        return expired

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


class ExpiredDetectionOverlay:
    def __init__(self, display_frames: int = 8) -> None:
        self.display_frames = max(0, int(display_frames))
        self._items: list[tuple[np.ndarray, int]] = []

    def add_tracks(self, tracks: list[TrackedDetection], frame_id: int) -> None:
        if self.display_frames == 0:
            return
        expire_at = frame_id + self.display_frames
        for track in tracks:
            self._items.append((track.values.astype(np.float32, copy=True), expire_at))

    def snapshot(self, frame_id: int) -> np.ndarray:
        self._items = [(values, expire_at) for values, expire_at in self._items if expire_at >= frame_id]
        if not self._items:
            return np.empty((0, 6), dtype=np.float32)
        return np.stack([values for values, _expire_at in self._items]).astype(np.float32)


class LatestFrameStream:
    """Read a network stream continuously and expose only the newest frame."""

    def __init__(
        self,
        url: str,
        read_timeout: float = 2.0,
        capture: Any | None = None,
    ) -> None:
        self.url = url
        self.read_timeout = max(0.1, float(read_timeout))
        self._cap = capture if capture is not None else cv2.VideoCapture(url)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._condition = threading.Condition()
        self._stop = False
        self._latest_frame: np.ndarray | None = None
        self._sequence = 0
        self._last_returned_sequence = 0
        self._read_failed = False
        self._thread: threading.Thread | None = None

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._reader_loop, name="latest-frame-stream", daemon=True)
        self._thread.start()

    def read(self) -> tuple[bool, np.ndarray | None]:
        deadline = time.perf_counter() + self.read_timeout
        with self._condition:
            while self._sequence == self._last_returned_sequence and not self._read_failed and not self._stop:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return False, None
                self._condition.wait(timeout=remaining)

            if self._read_failed and self._sequence == self._last_returned_sequence:
                return False, None
            if self._latest_frame is None:
                return False, None
            self._last_returned_sequence = self._sequence
            return True, self._latest_frame.copy()

    def release(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()

    def _reader_loop(self) -> None:
        while True:
            with self._condition:
                if self._stop:
                    return

            ok, frame = self._cap.read()
            with self._condition:
                if self._stop:
                    return
                if not ok or frame is None:
                    self._read_failed = True
                    self._condition.notify_all()
                    return
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8, copy=False)
                if frame.ndim != 3 or frame.shape[2] != 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                self._latest_frame = frame
                self._sequence += 1
                self._condition.notify_all()


class LatestJsonHub:
    def __init__(self, device_id: str) -> None:
        self._lock = threading.Lock()
        self._clients: set[socket.socket] = set()
        self._payload: dict[str, Any] = {
            "device_id": device_id,
            "status": "warming_up",
            "timestamp_ms": int(time.time() * 1000),
            "frame_id": 0,
            "image_size": None,
            "roi": None,
            "fps": 0.0,
            "inference_ms": 0.0,
            "detections": [],
        }

    def update(self, payload: dict[str, Any]) -> None:
        body: bytes
        with self._lock:
            self._payload = dict(payload)
            body = self._snapshot_bytes_locked()
        self.broadcast(body)

    def snapshot_bytes(self) -> bytes:
        with self._lock:
            return self._snapshot_bytes_locked()

    def add_ws_client(self, client: socket.socket) -> None:
        with self._lock:
            self._clients.add(client)

    def remove_ws_client(self, client: socket.socket) -> None:
        with self._lock:
            self._clients.discard(client)

    def broadcast(self, body: bytes | None = None) -> None:
        if body is None:
            body = self.snapshot_bytes()
        frame = websocket_text_frame(body)
        with self._lock:
            clients = list(self._clients)
        stale: list[socket.socket] = []
        for client in clients:
            try:
                client.sendall(frame)
            except OSError:
                stale.append(client)
        if stale:
            with self._lock:
                for client in stale:
                    self._clients.discard(client)
                    try:
                        client.close()
                    except OSError:
                        pass

    def _snapshot_bytes_locked(self) -> bytes:
        payload = dict(self._payload)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def websocket_text_frame(payload: bytes) -> bytes:
    length = len(payload)
    if length < 126:
        header = bytes((0x81, length))
    elif length <= 0xFFFF:
        header = bytes((0x81, 126)) + length.to_bytes(2, "big")
    else:
        header = bytes((0x81, 127)) + length.to_bytes(8, "big")
    return header + payload


class LatestJsonServer:
    def __init__(self, host: str, port: int, hub: LatestJsonHub) -> None:
        self.host = host
        self.port = int(port)
        self.hub = hub
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/latest.json"

    def start(self) -> None:
        if self._httpd is not None:
            return

        hub = self.hub

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/ws":
                    self._handle_websocket()
                    return
                if path == "/health":
                    self._send_bytes(b'{"status":"ok"}', "application/json; charset=utf-8")
                    return
                if path != "/latest.json":
                    self.send_error(404)
                    return
                self._send_bytes(hub.snapshot_bytes(), "application/json; charset=utf-8")

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _handle_websocket(self) -> None:
                key = self.headers.get("Sec-WebSocket-Key")
                upgrade = self.headers.get("Upgrade", "")
                if not key or upgrade.lower() != "websocket":
                    self.send_error(400)
                    return

                accept_source = (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
                accept = base64.b64encode(hashlib.sha1(accept_source).digest()).decode("ascii")
                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()

                client = self.connection
                hub.add_ws_client(client)
                try:
                    client.sendall(websocket_text_frame(hub.snapshot_bytes()))
                    while True:
                        data = client.recv(2)
                        if not data:
                            break
                        opcode = data[0] & 0x0F
                        masked = bool(data[1] & 0x80)
                        length = data[1] & 0x7F
                        if length == 126:
                            ext = client.recv(2)
                            if len(ext) < 2:
                                break
                            length = int.from_bytes(ext, "big")
                        elif length == 127:
                            ext = client.recv(8)
                            if len(ext) < 8:
                                break
                            length = int.from_bytes(ext, "big")
                        mask = client.recv(4) if masked else b""
                        remaining = length
                        while remaining > 0:
                            chunk = client.recv(min(remaining, 4096))
                            if not chunk:
                                remaining = 0
                                break
                            remaining -= len(chunk)
                        if opcode == 0x8:
                            break
                        if not mask and length > 0:
                            continue
                except OSError:
                    pass
                finally:
                    hub.remove_ws_client(client)

            def _send_bytes(self, body: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="latest-json-server",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._httpd = None
        self._thread = None


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


def box_center_in_roi(box: np.ndarray, roi: Roi, margin_pixels: int = 0) -> bool:
    margin = max(0, int(margin_pixels))
    center_x = (float(box[0]) + float(box[2])) / 2.0
    center_y = (float(box[1]) + float(box[3])) / 2.0
    return (
        roi.x1 - margin <= center_x <= roi.x2 + margin
        and roi.y1 - margin <= center_y <= roi.y2 + margin
    )


def ttl_frames_from_args(args: argparse.Namespace, fps: float) -> int:
    ttl_ms = max(0.0, float(getattr(args, "detection_ttl_ms", 0.0)))
    if ttl_ms <= 0.0:
        return 0
    effective_fps = fps if fps and fps > 1.0 else 25.0
    return max(1, int(math.ceil(effective_fps * ttl_ms / 1000.0)))


def cached_roi_due(
    detections: np.ndarray,
    now: float,
    last_refresh_time: float,
    interval_ms: float,
) -> bool:
    if detections.size == 0:
        return False
    interval = max(0.0, float(interval_ms))
    return interval > 0.0 and (now - last_refresh_time) * 1000.0 >= interval


class RefreshTileScheduler:
    def __init__(self, mode: str, interval_ms: float, tile_size: int) -> None:
        if mode not in {"full", "tiles"}:
            raise ValueError(f"Unknown full-frame refresh mode: {mode}")
        self.mode = mode
        self.interval_ms = max(0.0, float(interval_ms))
        self.tile_size = max(1, int(tile_size))
        self._next_due_time: float | None = None
        self._tile_index = 0

    def next_due_roi(self, frame_shape: tuple[int, int], now: float) -> Roi | None:
        if self.interval_ms <= 0.0:
            return None
        if self._next_due_time is None:
            self._next_due_time = now + self.interval_ms / 1000.0
            return None
        if now < self._next_due_time:
            return None

        h, w = frame_shape
        if self.mode == "full":
            self._next_due_time = now + self.interval_ms / 1000.0
            return Roi(0, 0, w, h, "full")

        tiles = refresh_tiles(frame_shape, self.tile_size)
        if not tiles:
            return None
        roi = tiles[self._tile_index % len(tiles)]
        self._tile_index += 1
        self._next_due_time = now + (self.interval_ms / len(tiles)) / 1000.0
        return roi


def refresh_tiles(frame_shape: tuple[int, int], tile_size: int) -> list[Roi]:
    frame_h, frame_w = frame_shape
    if frame_w <= 0 or frame_h <= 0:
        return []

    mid_x = min(max(1, frame_w // 2), frame_w)
    center_w = max(1, frame_w - mid_x)
    center_h = frame_h
    center_x1 = max(0, (frame_w - center_w) // 2)
    center_y1 = 0

    positions = [
        (0, 0, mid_x, frame_h),
        (mid_x, 0, frame_w, frame_h),
        (center_x1, center_y1, center_x1 + center_w, center_y1 + center_h),
    ]
    rois: list[Roi] = []
    seen: set[tuple[int, int, int, int]] = set()
    for x1, y1, x2, y2 in positions:
        x1 = min(max(0, int(x1)), frame_w)
        y1 = min(max(0, int(y1)), frame_h)
        x2 = min(max(x1, int(x2)), frame_w)
        y2 = min(max(y1, int(y2)), frame_h)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = Roi(x1, y1, x2, y2, "tile")
        key = (roi.x1, roi.y1, roi.x2, roi.y2)
        if key not in seen:
            rois.append(roi)
            seen.add(key)
    return rois


def union_roi_with_detections(
    roi: Roi,
    detections: np.ndarray,
    frame_shape: tuple[int, int],
    proximity_pixels: int = 0,
) -> Roi:
    if roi.reason == "full" or detections.size == 0:
        return roi

    frame_h, frame_w = frame_shape
    gap = max(0, int(proximity_pixels))
    nearby_mask = np.array(
        [box_center_in_roi(det[:4], roi, margin_pixels=gap) for det in detections],
        dtype=bool,
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


def detect(
    net: ncnn.Net,
    image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
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


def start_latest_json_server(args: argparse.Namespace) -> tuple[LatestJsonHub | None, LatestJsonServer | None]:
    if not args.latest_json:
        return None, None
    hub = LatestJsonHub(args.device_id)
    server = LatestJsonServer(args.latest_json_host, args.latest_json_port, hub)
    server.start()
    print(f"latest json: {server.url}")
    return hub, server


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


def draw_dashed_rectangle(
    image: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 8,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dash = max(2, int(dash_length))
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_expired_detections(
    image: np.ndarray,
    expired_detections: np.ndarray,
    text_renderer: TextRenderer,
) -> np.ndarray:
    if expired_detections.size == 0:
        return image
    out = image.copy()
    color = (0, 0, 255)
    for x1, y1, x2, y2, _score, _class_id in expired_detections:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        draw_dashed_rectangle(out, p1, p2, color, thickness=1, dash_length=8)
        out = text_renderer.put_text(out, "TTL expired", (p1[0], max(14, p1[1] - 4)), color, scale=0.45, thickness=1)
    return out


def draw_status(
    image: np.ndarray,
    text_renderer: TextRenderer,
    fps: float,
    inference_ms: float,
    roi: Roi | None,
    detections: np.ndarray,
    skipped: bool,
    mode: str = "compact",
    ttl_frames: int | None = None,
    ttl_ms: float | None = None,
    refresh_mode: str | None = None,
) -> np.ndarray:
    if mode == "off":
        return image

    status_text = "Status: Sleep (No Motion)" if skipped else "Status: Detecting"
    status_color = (150, 150, 150) if skipped else (0, 0, 255)

    if roi is not None:
        cv2.rectangle(image, (roi.x1, roi.y1), (roi.x2, roi.y2), ROI_COLOR, 2)

    if mode == "compact":
        roi_text = roi.reason if roi is not None else "none"
        infer_text = f"{inference_ms:.0f}ms" if not skipped else "skip"
        roi_box = (
            f"{roi.x1},{roi.y1},{roi.x2},{roi.y2}" if roi is not None else "-"
        )
        if ttl_frames is not None:
            ttl_text = f"{ttl_frames}f"
        elif ttl_ms is not None:
            ttl_text = f"{ttl_ms:.0f}ms"
        else:
            ttl_text = "-"
        lines = [
            status_text.replace("Status: ", ""),
            f"FPS: {fps:.1f}",
            f"Infer: {infer_text}",
            f"Detections: {len(detections)}",
            f"ROI: {roi_text}",
            f"Box: {roi_box}",
            f"TTL: {ttl_text}",
            f"Refresh: {refresh_mode or '-'}",
        ]
        text_scale = 0.48
        line_height = 18
        max_text_width = 0
        for line in lines:
            (tw, _th), _baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
            max_text_width = max(max_text_width, tw)
        panel_width = min(max(150, max_text_width + 20), min(300, max(120, image.shape[1] - 16)))
        panel_height = min(10 + line_height * len(lines), max(30, image.shape[0] - 16))
        overlay = image.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        y = 26
        for idx, line in enumerate(lines):
            color = status_color if idx == 0 else (255, 255, 255)
            image = text_renderer.put_text(image, line, (16, y), color, scale=text_scale, thickness=1)
            y += line_height
            if y > 8 + panel_height - 4:
                break
        return image

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


def render_diagnostic_frame(
    frame: np.ndarray,
    detections: np.ndarray,
    labels: list[str],
    text_renderer: TextRenderer,
    expired_detections: np.ndarray,
    fps: float,
    inference_ms: float,
    roi: Roi | None,
    skipped: bool,
    args: argparse.Namespace,
    current_ttl_frames: int,
) -> np.ndarray:
    rendered = draw_detections(frame, detections, labels, text_renderer)
    rendered = draw_expired_detections(rendered, expired_detections, text_renderer)
    return draw_status(
        rendered,
        text_renderer,
        fps,
        inference_ms,
        roi,
        detections,
        skipped=skipped,
        mode=args.status_overlay,
        ttl_frames=current_ttl_frames,
        ttl_ms=args.detection_ttl_ms,
        refresh_mode=args.full_frame_refresh_mode,
    )


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
    rendered = draw_status(
        rendered,
        TextRenderer(),
        0.0,
        elapsed_ms,
        full_roi,
        detections,
        skipped=False,
        mode=args.status_overlay,
        ttl_frames=ttl_frames_from_args(args, 0.0),
        ttl_ms=args.detection_ttl_ms,
        refresh_mode=args.full_frame_refresh_mode,
    )
    if not cv2.imwrite(str(output_path), rendered):
        raise RuntimeError(f"Could not write output image: {output_path}")

    print(f"image: {args.image}")
    print(f"output: {output_path}")
    print(f"detections: {len(detections)}")
    print(f"inference: {elapsed_ms:.2f} ms")


def run_video(args: argparse.Namespace, net: ncnn.Net, labels: list[str]) -> None:
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    source_fps = video.get(cv2.CAP_PROP_FPS)
    output_fps = source_fps if source_fps and source_fps > 0 else 25.0
    writer: cv2.VideoWriter | None = None
    output_path = args.output

    roi_selector = MotionRoiSelector(
        mode=args.roi_mode,
        min_area=args.roi_min_area,
        padding=args.roi_padding,
        padding_pixels=args.roi_padding_pixels,
        history=args.mog2_history,
        var_threshold=args.mog2_var_threshold,
        hold_frames=args.roi_hold_frames,
        smooth_alpha=args.roi_smooth_alpha,
        motion_source=args.motion_source,
        frame_diff_threshold=args.frame_diff_threshold,
        frame_diff_min_area=args.frame_diff_min_area,
        frame_diff_alpha=args.frame_diff_alpha,
    )
    backend = BackendClient(args.backend_url, timeout=args.backend_timeout, max_queue=args.backend_queue)
    latest_json_hub, latest_json_server = start_latest_json_server(args)
    text_renderer = TextRenderer()
    fps_meter = FpsMeter()
    detection_cache = DetectionCache(
        ttl_frames=ttl_frames_from_args(args, output_fps),
        dedupe_iou=args.iou_thres,
    )
    expired_overlay = ExpiredDetectionOverlay(args.expired_ttl_display_frames)
    refresh_scheduler = RefreshTileScheduler(
        args.full_frame_refresh_mode,
        args.full_frame_refresh_ms,
        args.img_size,
    )
    frame_id = 0
    last_inference_ms = 0.0
    last_cached_roi_time = 0.0
    render_output = output_path is not None or not args.headless

    if not args.headless:
        print("Press q or Esc to quit.")

    try:
        while True:
            ok, frame = video.read()
            if not ok or frame is None:
                break

            frame_id += 1
            now = time.perf_counter()
            fps = fps_meter.update()
            roi = roi_selector.select(frame, frame_id)
            current_ttl_frames = ttl_frames_from_args(args, fps or output_fps)
            detection_cache.set_ttl_frames(current_ttl_frames)
            active_detections = detection_cache.active_detections(frame_id)

            if roi is not None:
                roi = union_roi_with_detections(
                    roi,
                    active_detections,
                    frame.shape[:2],
                    proximity_pixels=args.roi_padding_pixels,
                )
            elif cached_roi_due(
                active_detections,
                now,
                last_cached_roi_time,
                args.cached_roi_ms,
            ):
                roi = cached_roi_from_detections(
                    active_detections,
                    frame.shape[:2],
                    args.cached_roi_padding_pixels,
                    args.roi_min_size_pixels,
                )
            if roi is None and args.roi_mode == "hybrid":
                roi = refresh_scheduler.next_due_roi(frame.shape[:2], now)
            if roi is not None:
                roi = expand_roi_to_min_size(roi, frame.shape[:2], args.roi_min_size_pixels)
            skipped = roi is None

            if roi is not None:
                raw_detections, inference_ms = detect_roi(net, frame, roi, args)
                if roi.reason == "full":
                    detection_cache.update_full(raw_detections, frame_id)
                else:
                    detection_cache.update_region(raw_detections, frame_id, roi)
                if roi.reason == "cached":
                    last_cached_roi_time = now
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
                if latest_json_hub is not None:
                    latest_json_hub.update(payload)
                backend.submit(payload)
            else:
                detections, _cached_flags = detection_cache.snapshot(frame_id)
                inference_ms = last_inference_ms

            if args.show_expired_ttl:
                expired_overlay.add_tracks(detection_cache.pop_expired_tracks(), frame_id)
                expired_detections = expired_overlay.snapshot(frame_id)
            else:
                detection_cache.pop_expired_tracks()
                expired_detections = np.empty((0, 6), dtype=np.float32)

            should_quit = False
            if render_output:
                rendered = render_diagnostic_frame(
                    frame,
                    detections,
                    labels,
                    text_renderer,
                    expired_detections,
                    fps,
                    inference_ms,
                    roi,
                    skipped,
                    args,
                    current_ttl_frames,
                )

                if output_path is not None:
                    if writer is None:
                        h, w = rendered.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not write output video: {output_path}")
                    writer.write(rendered)

                if not args.headless:
                    cv2.imshow("ncnn yolov5 video", rendered)
                    key = cv2.waitKey(1) & 0xFF
                    should_quit = key in (27, ord("q"))

            if should_quit:
                break
    finally:
        if latest_json_server is not None:
            latest_json_server.close()
        backend.close()
        video.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"video: {args.video}")
    if output_path is not None:
        print(f"output: {output_path}")
    print(f"frames: {frame_id}")


def run_stream(args: argparse.Namespace, net: ncnn.Net, labels: list[str]) -> None:
    stream = LatestFrameStream(args.stream_url)
    if not stream.is_opened():
        raise RuntimeError(f"Could not open stream URL: {args.stream_url}")
    stream.start()

    source_fps = stream.fps
    output_fps = source_fps if source_fps and source_fps > 0 else 25.0
    writer: cv2.VideoWriter | None = None
    output_path = args.output

    roi_selector = MotionRoiSelector(
        mode=args.roi_mode,
        min_area=args.roi_min_area,
        padding=args.roi_padding,
        padding_pixels=args.roi_padding_pixels,
        history=args.mog2_history,
        var_threshold=args.mog2_var_threshold,
        hold_frames=args.roi_hold_frames,
        smooth_alpha=args.roi_smooth_alpha,
        motion_source=args.motion_source,
        frame_diff_threshold=args.frame_diff_threshold,
        frame_diff_min_area=args.frame_diff_min_area,
        frame_diff_alpha=args.frame_diff_alpha,
    )
    backend = BackendClient(args.backend_url, timeout=args.backend_timeout, max_queue=args.backend_queue)
    latest_json_hub, latest_json_server = start_latest_json_server(args)
    text_renderer = TextRenderer()
    fps_meter = FpsMeter()
    detection_cache = DetectionCache(
        ttl_frames=ttl_frames_from_args(args, output_fps),
        dedupe_iou=args.iou_thres,
    )
    expired_overlay = ExpiredDetectionOverlay(args.expired_ttl_display_frames)
    refresh_scheduler = RefreshTileScheduler(
        args.full_frame_refresh_mode,
        args.full_frame_refresh_ms,
        args.img_size,
    )
    frame_id = 0
    last_inference_ms = 0.0
    last_cached_roi_time = 0.0
    render_output = output_path is not None or not args.headless

    if not args.headless:
        print("Press q or Esc to quit.")

    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                break

            frame_id += 1
            now = time.perf_counter()
            fps = fps_meter.update()
            roi = roi_selector.select(frame, frame_id)
            current_ttl_frames = ttl_frames_from_args(args, fps or output_fps)
            detection_cache.set_ttl_frames(current_ttl_frames)
            active_detections = detection_cache.active_detections(frame_id)

            if roi is not None:
                roi = union_roi_with_detections(
                    roi,
                    active_detections,
                    frame.shape[:2],
                    proximity_pixels=args.roi_padding_pixels,
                )
            elif cached_roi_due(
                active_detections,
                now,
                last_cached_roi_time,
                args.cached_roi_ms,
            ):
                roi = cached_roi_from_detections(
                    active_detections,
                    frame.shape[:2],
                    args.cached_roi_padding_pixels,
                    args.roi_min_size_pixels,
                )
            if roi is None and args.roi_mode == "hybrid":
                roi = refresh_scheduler.next_due_roi(frame.shape[:2], now)
            if roi is not None:
                roi = expand_roi_to_min_size(roi, frame.shape[:2], args.roi_min_size_pixels)
            skipped = roi is None

            if roi is not None:
                raw_detections, inference_ms = detect_roi(net, frame, roi, args)
                if roi.reason == "full":
                    detection_cache.update_full(raw_detections, frame_id)
                else:
                    detection_cache.update_region(raw_detections, frame_id, roi)
                if roi.reason == "cached":
                    last_cached_roi_time = now
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
                if latest_json_hub is not None:
                    latest_json_hub.update(payload)
                backend.submit(payload)
            else:
                detections, _cached_flags = detection_cache.snapshot(frame_id)
                inference_ms = last_inference_ms

            if args.show_expired_ttl:
                expired_overlay.add_tracks(detection_cache.pop_expired_tracks(), frame_id)
                expired_detections = expired_overlay.snapshot(frame_id)
            else:
                detection_cache.pop_expired_tracks()
                expired_detections = np.empty((0, 6), dtype=np.float32)

            should_quit = False
            if render_output:
                rendered = render_diagnostic_frame(
                    frame,
                    detections,
                    labels,
                    text_renderer,
                    expired_detections,
                    fps,
                    inference_ms,
                    roi,
                    skipped,
                    args,
                    current_ttl_frames,
                )

                if output_path is not None:
                    if writer is None:
                        h, w = rendered.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not write output video: {output_path}")
                    writer.write(rendered)

                if not args.headless:
                    cv2.imshow("ncnn yolov5 stream", rendered)
                    key = cv2.waitKey(1) & 0xFF
                    should_quit = key in (27, ord("q"))

            if should_quit:
                break
    finally:
        if latest_json_server is not None:
            latest_json_server.close()
        backend.close()
        stream.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"stream: {args.stream_url}")
    if output_path is not None:
        print(f"output: {output_path}")
    print(f"frames: {frame_id}")


def run_camera(args: argparse.Namespace, net: ncnn.Net, labels: list[str]) -> None:
    camera = CameraInput(
        camera_index=args.camera_index,
        width=args.camera_width,
        height=args.camera_height,
    )
    if not camera.is_opened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    roi_selector = MotionRoiSelector(
        mode=args.roi_mode,
        min_area=args.roi_min_area,
        padding=args.roi_padding,
        padding_pixels=args.roi_padding_pixels,
        history=args.mog2_history,
        var_threshold=args.mog2_var_threshold,
        hold_frames=args.roi_hold_frames,
        smooth_alpha=args.roi_smooth_alpha,
        motion_source=args.motion_source,
        frame_diff_threshold=args.frame_diff_threshold,
        frame_diff_min_area=args.frame_diff_min_area,
        frame_diff_alpha=args.frame_diff_alpha,
    )
    backend = BackendClient(args.backend_url, timeout=args.backend_timeout, max_queue=args.backend_queue)
    latest_json_hub, latest_json_server = start_latest_json_server(args)
    text_renderer = TextRenderer()
    fps_meter = FpsMeter()
    detection_cache = DetectionCache(ttl_frames=ttl_frames_from_args(args, 0.0), dedupe_iou=args.iou_thres)
    expired_overlay = ExpiredDetectionOverlay(args.expired_ttl_display_frames)
    refresh_scheduler = RefreshTileScheduler(
        args.full_frame_refresh_mode,
        args.full_frame_refresh_ms,
        args.img_size,
    )
    frame_id = 0
    last_inference_ms = 0.0
    last_cached_roi_time = 0.0

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
            current_ttl_frames = ttl_frames_from_args(args, fps)
            detection_cache.set_ttl_frames(current_ttl_frames)
            active_detections = detection_cache.active_detections(frame_id)

            if roi is not None:
                roi = union_roi_with_detections(
                    roi,
                    active_detections,
                    frame.shape[:2],
                    proximity_pixels=args.roi_padding_pixels,
                )
            elif cached_roi_due(
                active_detections,
                now,
                last_cached_roi_time,
                args.cached_roi_ms,
            ):
                roi = cached_roi_from_detections(
                    active_detections,
                    frame.shape[:2],
                    args.cached_roi_padding_pixels,
                    args.roi_min_size_pixels,
                )
            if roi is None and args.roi_mode == "hybrid":
                roi = refresh_scheduler.next_due_roi(frame.shape[:2], now)
            if roi is not None:
                roi = expand_roi_to_min_size(roi, frame.shape[:2], args.roi_min_size_pixels)
            skipped = roi is None

            if roi is not None:
                raw_detections, inference_ms = detect_roi(net, frame, roi, args)
                if roi.reason == "full":
                    detection_cache.update_full(raw_detections, frame_id)
                else:
                    detection_cache.update_region(raw_detections, frame_id, roi)
                if roi.reason == "cached":
                    last_cached_roi_time = now
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
                if latest_json_hub is not None:
                    latest_json_hub.update(payload)
                backend.submit(payload)
            else:
                detections, cached_flags = detection_cache.snapshot(frame_id)
                inference_ms = last_inference_ms

            if args.show_expired_ttl:
                expired_overlay.add_tracks(detection_cache.pop_expired_tracks(), frame_id)
                expired_detections = expired_overlay.snapshot(frame_id)
            else:
                detection_cache.pop_expired_tracks()
                expired_detections = np.empty((0, 6), dtype=np.float32)

            if args.headless:
                continue

            rendered = render_diagnostic_frame(
                frame,
                detections,
                labels,
                text_renderer,
                expired_detections,
                fps,
                inference_ms,
                roi,
                skipped,
                args,
                current_ttl_frames,
            )
            cv2.imshow("ncnn yolov5 camera", rendered)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        if latest_json_server is not None:
            latest_json_server.close()
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
    parser.add_argument("--video", type=Path, help="Video path for file-based inference.")
    parser.add_argument(
        "--stream-url",
        default=DEFAULT_STREAM_URL,
        help="Network video stream URL, for example an mjpg-streamer URL.",
    )
    parser.add_argument("--output", type=Path, help="Output image or video path.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=640)
    parser.add_argument("--headless", action="store_true", help="Do not open an OpenCV preview window.")
    parser.add_argument("--self-test", action="store_true", help="Run one inference on a blank image.")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--input-name", default="in0")
    parser.add_argument("--output-name", default="out0")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--nms-mode", choices=("class_aware", "class_agnostic"), default="class_agnostic")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--roi-mode", choices=("hybrid", "roi", "full"), default="full")
    parser.add_argument("--full-frame-refresh-mode", choices=("full", "tiles"), default="tiles")
    parser.add_argument("--full-frame-refresh-ms", type=float, default=0.0)
    parser.add_argument("--roi-min-area", type=int, default=800)
    parser.add_argument("--roi-min-size-pixels", type=int, default=96)
    parser.add_argument("--roi-padding", type=float, default=0.15)
    parser.add_argument("--roi-padding-pixels", type=int, default=50)
    parser.add_argument("--roi-hold-frames", type=int, default=0)
    parser.add_argument("--roi-smooth-alpha", type=float, default=0.6)
    parser.add_argument("--cached-roi-ms", type=float, default=0.0)
    parser.add_argument("--cached-roi-padding-pixels", type=int, default=60)
    parser.add_argument("--detection-ttl-ms", type=float, default=0.0)
    parser.add_argument("--show-expired-ttl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expired-ttl-display-frames", type=int, default=8)
    parser.add_argument("--status-overlay", choices=("compact", "full", "off"), default="compact")
    parser.add_argument("--motion-source", choices=("frame_diff", "mog2", "both"), default="mog2")
    parser.add_argument("--mog2-history", type=int, default=80)
    parser.add_argument("--mog2-var-threshold", type=float, default=25.0)
    parser.add_argument("--frame-diff-threshold", type=int, default=12)
    parser.add_argument("--frame-diff-min-area", type=int, default=300)
    parser.add_argument("--frame-diff-alpha", type=float, default=0.08)
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--backend-timeout", type=float, default=1.0)
    parser.add_argument("--backend-queue", type=int, default=8)
    parser.add_argument(
        "--latest-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose the latest detection payload at /latest.json.",
    )
    parser.add_argument("--latest-json-host", default=DEFAULT_LATEST_JSON_HOST)
    parser.add_argument("--latest-json-port", type=int, default=DEFAULT_LATEST_JSON_PORT)
    parser.add_argument("--device-id", default="pi-camera-01")
    return parser.parse_args()


def cli_option_provided(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def main() -> None:
    args = parse_args()
    explicit_stream_url = cli_option_provided(sys.argv[1:], "--stream-url")
    input_modes = sum(
        bool(value)
        for value in (
            args.self_test,
            args.image is not None,
            args.video is not None,
            explicit_stream_url,
        )
    )
    if input_modes > 1:
        raise SystemExit("Choose only one input mode: --self-test, --image, --video, or --stream-url.")

    labels = read_labels(args.labels)
    net = load_net(args)

    if args.self_test:
        run_self_test(args, net)
    elif args.image is not None:
        run_image(args, net, labels)
    elif args.video is not None:
        run_video(args, net, labels)
    elif args.stream_url is not None:
        run_stream(args, net, labels)
    else:
        run_camera(args, net, labels)


if __name__ == "__main__":
    main()
