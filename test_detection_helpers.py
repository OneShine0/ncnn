from __future__ import annotations

import argparse
import sys
import types
import unittest

import numpy as np


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0

    @staticmethod
    def getTextSize(text: str, _font: int, scale: float, thickness: int) -> tuple[tuple[int, int], int]:
        return (int(len(text) * 10 * scale), int(18 * scale + thickness)), 2

    @staticmethod
    def rectangle(image: np.ndarray, _p1: tuple[int, int], _p2: tuple[int, int], color: tuple[int, int, int], _thickness: int) -> np.ndarray:
        if image.size:
            image[0, 0] = color
        return image

    @staticmethod
    def addWeighted(src1: np.ndarray, alpha: float, src2: np.ndarray, beta: float, gamma: float, dst: np.ndarray) -> np.ndarray:
        dst[:] = np.clip(src1 * alpha + src2 * beta + gamma, 0, 255).astype(dst.dtype)
        return dst


class SimpleTextRenderer:
    def put_text(
        self,
        image: np.ndarray,
        _text: str,
        _origin: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.6,
        thickness: int = 1,
    ) -> np.ndarray:
        if image.size:
            image[-1, -1] = color
        return image


sys.modules.setdefault("cv2", FakeCv2())
sys.modules.setdefault("ncnn", types.SimpleNamespace())

import detect_ncnn_yolov5 as detector


class DetectionHelperTests(unittest.TestCase):
    def assert_first_four_tiles_cover_frame(self, tiles: list[detector.Roi], frame_shape: tuple[int, int]) -> None:
        frame_h, frame_w = frame_shape
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        for tile in tiles[:4]:
            mask[tile.y1 : tile.y2, tile.x1 : tile.x2] = 1
        self.assertTrue(np.all(mask == 1))

    def test_refresh_tiles_cover_640_square_with_center(self) -> None:
        tiles = detector.refresh_tiles((640, 640), 320)
        self.assertEqual(
            [(tile.x1, tile.y1, tile.x2, tile.y2, tile.reason) for tile in tiles],
            [
                (0, 0, 320, 320, "tile"),
                (320, 0, 640, 320, "tile"),
                (0, 320, 320, 640, "tile"),
                (320, 320, 640, 640, "tile"),
                (160, 160, 480, 480, "tile"),
            ],
        )
        self.assert_first_four_tiles_cover_frame(tiles, (640, 640))

    def test_refresh_tiles_cover_wide_frame_without_gap(self) -> None:
        tiles = detector.refresh_tiles((480, 640), 320)
        self.assertEqual(
            [(tile.x1, tile.y1, tile.x2, tile.y2, tile.reason) for tile in tiles],
            [
                (0, 0, 320, 240, "tile"),
                (320, 0, 640, 240, "tile"),
                (0, 240, 320, 480, "tile"),
                (320, 240, 640, 480, "tile"),
                (160, 120, 480, 360, "tile"),
            ],
        )
        self.assert_first_four_tiles_cover_frame(tiles, (480, 640))

    def test_refresh_tiles_cover_odd_frame_without_gap(self) -> None:
        tiles = detector.refresh_tiles((481, 641), 320)
        self.assertEqual(
            [(tile.x1, tile.y1, tile.x2, tile.y2, tile.reason) for tile in tiles],
            [
                (0, 0, 320, 240, "tile"),
                (320, 0, 641, 240, "tile"),
                (0, 240, 320, 481, "tile"),
                (320, 240, 641, 481, "tile"),
                (160, 120, 481, 361, "tile"),
            ],
        )
        self.assert_first_four_tiles_cover_frame(tiles, (481, 641))
        widths = [tile.width for tile in tiles[:4]]
        heights = [tile.height for tile in tiles[:4]]
        self.assertLessEqual(max(widths) - min(widths), 1)
        self.assertLessEqual(max(heights) - min(heights), 1)

    def test_tile_scheduler_spreads_one_cycle_across_tiles(self) -> None:
        scheduler = detector.RefreshTileScheduler("tiles", interval_ms=2000, tile_size=320)
        self.assertIsNone(scheduler.next_due_roi((640, 640), 10.0))
        self.assertEqual(scheduler.next_due_roi((640, 640), 12.0), detector.Roi(0, 0, 320, 320, "tile"))
        self.assertIsNone(scheduler.next_due_roi((640, 640), 12.2))
        self.assertEqual(scheduler.next_due_roi((640, 640), 12.4), detector.Roi(320, 0, 640, 320, "tile"))

    def test_ttl_ms_uses_runtime_fps_unless_frames_are_explicit(self) -> None:
        args = argparse.Namespace(detection_ttl_frames=None, detection_ttl_ms=500.0, camera_fps=None)
        self.assertEqual(detector.ttl_frames_from_args(args, 30.0), 15)

        args.detection_ttl_frames = 0
        self.assertEqual(detector.ttl_frames_from_args(args, 30.0), 0)

    def test_region_update_clears_checked_tile_tracks(self) -> None:
        cache = detector.DetectionCache(ttl_frames=0)
        cache.update_motion(
            np.array(
                [
                    [20, 20, 60, 60, 0.9, 0],
                    [420, 420, 460, 460, 0.8, 1],
                ],
                dtype=np.float32,
            ),
            frame_id=1,
        )
        cache.update_region(np.empty((0, 6), dtype=np.float32), frame_id=2, roi=detector.Roi(0, 0, 320, 320, "tile"))

        detections, cached_flags = cache.snapshot(frame_id=2)
        self.assertEqual(detections.shape[0], 1)
        self.assertEqual(cached_flags, [True])
        self.assertEqual(int(detections[0, 0]), 420)

    def test_status_overlay_modes_are_callable(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = np.empty((0, 6), dtype=np.float32)
        renderer = SimpleTextRenderer()
        roi = detector.Roi(0, 0, 160, 120, "tile")

        compact = detector.draw_status(
            frame.copy(),
            renderer,
            fps=25.0,
            inference_ms=12.3,
            roi=roi,
            detections=detections,
            skipped=False,
            mode="compact",
            ttl_frames=13,
            ttl_ms=500.0,
            refresh_mode="tiles",
        )
        full = detector.draw_status(
            frame.copy(),
            renderer,
            fps=25.0,
            inference_ms=12.3,
            roi=roi,
            detections=detections,
            skipped=False,
            mode="full",
        )
        off_source = frame.copy()
        off = detector.draw_status(
            off_source,
            renderer,
            fps=25.0,
            inference_ms=12.3,
            roi=roi,
            detections=detections,
            skipped=False,
            mode="off",
        )

        self.assertEqual(compact.shape, frame.shape)
        self.assertEqual(full.shape, frame.shape)
        self.assertIs(off, off_source)


if __name__ == "__main__":
    unittest.main()
