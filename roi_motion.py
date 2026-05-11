from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Roi:
    x1: int
    y1: int
    x2: int
    y2: int
    reason: str

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y1 : self.y2, self.x1 : self.x2]

    def to_dict(self) -> dict[str, int | str]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "reason": self.reason,
        }


class MotionRoiSelector:
    def __init__(
        self,
        mode: str = "roi",
        full_frame_interval: int = 30,
        min_area: int = 2000,
        padding: float = 0.15,
        padding_pixels: int = 30,
        history: int = 500,
        var_threshold: float = 50.0,
        hold_frames: int = 0,
        smooth_alpha: float = 0.6,
        motion_source: str = "mog2",
        frame_diff_enabled: bool = True,
        frame_diff_threshold: int = 12,
        frame_diff_min_area: int = 300,
        frame_diff_alpha: float = 0.08,
    ) -> None:
        if mode not in {"hybrid", "roi", "full"}:
            raise ValueError(f"Unknown ROI mode: {mode}")
        if motion_source not in {"frame_diff", "mog2", "both"}:
            raise ValueError(f"Unknown motion source: {motion_source}")
        self.mode = mode
        self.full_frame_interval = max(0, full_frame_interval)
        self.min_area = max(1, min_area)
        self.padding = max(0.0, padding)
        self.padding_pixels = max(0, int(padding_pixels))
        self.hold_frames = max(0, hold_frames)
        self.smooth_alpha = min(max(smooth_alpha, 0.0), 1.0)
        self.motion_source = motion_source
        self.frame_diff_enabled = bool(frame_diff_enabled)
        self.frame_diff_threshold = max(1, int(frame_diff_threshold))
        self.frame_diff_min_area = max(1, int(frame_diff_min_area))
        self.frame_diff_alpha = min(max(frame_diff_alpha, 0.0), 1.0)
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._held_roi: Roi | None = None
        self._hold_remaining = 0
        self._diff_reference: np.ndarray | None = None

    def select(self, frame: np.ndarray, frame_id: int) -> Roi | None:
        h, w = frame.shape[:2]
        full_due = self.mode == "full" or frame_id == 1
        if (
            self.mode == "hybrid"
            and self.full_frame_interval > 0
            and frame_id % self.full_frame_interval == 0
        ):
            full_due = True

        motion_roi = None
        if self.mode in {"hybrid", "roi"}:
            motion_roi = self._motion_roi(frame)

        if full_due:
            return Roi(0, 0, w, h, "full")
        if motion_roi is not None:
            stable_roi = self._stabilize_motion_roi(motion_roi, frame.shape[:2])
            self._held_roi = stable_roi
            self._hold_remaining = self.hold_frames
            return stable_roi

        if self._held_roi is not None and self._hold_remaining > 0:
            self._hold_remaining -= 1
            return self._held_roi
        return None

    def _motion_roi(self, frame: np.ndarray) -> Roi | None:
        boxes: list[tuple[int, int, int, int]] = []

        if self.motion_source in {"mog2", "both"}:
            mask = self._subtractor.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, x + w, y + h))

        if self.frame_diff_enabled and self.motion_source in {"frame_diff", "both"}:
            boxes.extend(self._frame_diff_boxes(frame))

        if not boxes:
            return None

        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)

        frame_h, frame_w = frame.shape[:2]
        if self.padding_pixels > 0:
            pad_x = self.padding_pixels
            pad_y = self.padding_pixels
        else:
            pad_x = int(round((x2 - x1) * self.padding))
            pad_y = int(round((y2 - y1) * self.padding))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_w, x2 + pad_x)
        y2 = min(frame_h, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None
        return Roi(x1, y1, x2, y2, "motion")

    def _frame_diff_boxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_f = gray.astype(np.float32)
        if self._diff_reference is None:
            self._diff_reference = gray_f
            return []

        reference_u8 = cv2.convertScaleAbs(self._diff_reference)
        diff = cv2.absdiff(gray, reference_u8)
        _, mask = cv2.threshold(diff, self.frame_diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)

        boxes: list[tuple[int, int, int, int]] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.frame_diff_min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x + w, y + h))

        if self.frame_diff_alpha > 0.0:
            cv2.accumulateWeighted(gray_f, self._diff_reference, self.frame_diff_alpha)
        return boxes

    def _stabilize_motion_roi(self, roi: Roi, frame_shape: tuple[int, int]) -> Roi:
        if self._held_roi is None or self.smooth_alpha <= 0.0:
            return roi

        alpha = self.smooth_alpha
        x1 = int(round(self._held_roi.x1 * alpha + roi.x1 * (1.0 - alpha)))
        y1 = int(round(self._held_roi.y1 * alpha + roi.y1 * (1.0 - alpha)))
        x2 = int(round(self._held_roi.x2 * alpha + roi.x2 * (1.0 - alpha)))
        y2 = int(round(self._held_roi.y2 * alpha + roi.y2 * (1.0 - alpha)))

        frame_h, frame_w = frame_shape
        x1 = min(max(0, x1), frame_w - 1)
        y1 = min(max(0, y1), frame_h - 1)
        x2 = min(max(x1 + 1, x2), frame_w)
        y2 = min(max(y1 + 1, y2), frame_h)
        return Roi(x1, y1, x2, y2, roi.reason)
