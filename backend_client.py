from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any


class BackendClient:
    """Small async HTTP JSON sender.

    The camera loop can keep running even when the backend is slow or offline.
    This uses only Python's standard library to keep Raspberry Pi setup light.
    """

    def __init__(self, url: str | None, timeout: float = 1.0, max_queue: int = 8) -> None:
        self.url = url
        self.timeout = timeout
        self.enabled = bool(url)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error_log = 0.0

        if self.enabled:
            self._thread = threading.Thread(target=self._worker, name="backend-client", daemon=True)
            self._thread.start()

    def submit(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                pass

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _worker(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._post_json(payload)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                now = time.monotonic()
                if now - self._last_error_log > 5.0:
                    print(f"backend post failed: {exc}")
                    self._last_error_log = now
            finally:
                self._queue.task_done()

    def _post_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response.read()
