from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np


class RawFrameHub:
    def __init__(self, fps: float = 10.0, width: int = 640, quality: int = 70) -> None:
        self.fps = max(0.0, float(fps))
        self.width = max(0, int(width))
        self.quality = min(max(1, int(quality)), 100)
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._last_encode_time = 0.0

    def update(self, frame: np.ndarray, now: float | None = None) -> bool:
        if frame is None or frame.size == 0:
            return False

        current_time = time.monotonic() if now is None else float(now)
        min_interval = 1.0 / self.fps if self.fps > 0.0 else 0.0
        with self._condition:
            if (
                self._jpeg is not None
                and min_interval > 0.0
                and current_time - self._last_encode_time < min_interval
            ):
                return False

        encoded_frame = self._prepare_frame(frame)
        ok, encoded = cv2.imencode(
            ".jpg",
            encoded_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
        )
        if not ok:
            return False

        with self._condition:
            self._jpeg = encoded.tobytes()
            self._sequence += 1
            self._last_encode_time = current_time
            self._condition.notify_all()
        return True

    def snapshot(self) -> bytes | None:
        with self._condition:
            return self._jpeg

    def wait_for_frame(self, after_sequence: int = 0, timeout: float = 1.0) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._jpeg is None or self._sequence <= after_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)
            return self._sequence, self._jpeg

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        source = frame
        if self.width > 0 and frame.shape[1] > self.width:
            scale = self.width / float(frame.shape[1])
            target_h = max(1, int(round(frame.shape[0] * scale)))
            source = cv2.resize(frame, (self.width, target_h), interpolation=cv2.INTER_AREA)
        return source


class _RawMjpegHandler(BaseHTTPRequestHandler):
    server: "_RawMjpegHttpServer"

    def do_GET(self) -> None:
        if self.path not in {"/stream.mjpg", "/"}:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        sequence = 0
        try:
            while not self.server.stop_event.is_set():
                item = self.server.hub.wait_for_frame(sequence, timeout=1.0)
                if item is None:
                    continue
                sequence, jpeg = item
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class _RawMjpegHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], hub: RawFrameHub) -> None:
        super().__init__(server_address, _RawMjpegHandler)
        self.hub = hub
        self.stop_event = threading.Event()


class RawMjpegServer:
    def __init__(self, host: str, port: int, hub: RawFrameHub) -> None:
        self.host = host
        self.port = int(port)
        self.hub = hub
        self._server = _RawMjpegHttpServer((self.host, self.port), self.hub)
        self._thread = threading.Thread(target=self._server.serve_forever, name="raw-mjpeg-server", daemon=True)

    @property
    def url(self) -> str:
        display_host = "127.0.0.1" if self.host in {"", "0.0.0.0"} else self.host
        return f"http://{display_host}:{self.port}/stream.mjpg"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.stop_event.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)
