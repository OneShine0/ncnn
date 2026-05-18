from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraInput:
    """Camera frame source used by the detector loop.

    Raspberry Pi ports should keep the same public methods and return BGR
    uint8 frames shaped as H x W x 3.
    """

    camera_index: int = 0
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        if os.name == "nt":
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.camera_index)

        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if frame.ndim != 3 or frame.shape[2] != 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return True, frame

    def release(self) -> None:
        self._cap.release()
