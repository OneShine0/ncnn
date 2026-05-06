from __future__ import annotations

import argparse
import sys
import types
import unittest

import numpy as np


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


sys.modules.setdefault("ncnn", types.SimpleNamespace())

import detect_ncnn_yolov5 as detector


class DetectionHelperTests(unittest.TestCase):
    def assert_first_two_tiles_cover_frame(self, tiles: list[detector.Roi], frame_shape: tuple[int, int]) -> None:
        frame_h, frame_w = frame_shape
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        for tile in tiles[:2]:
            mask[tile.y1 : tile.y2, tile.x1 : tile.x2] = 1
        self.assertTrue(np.all(mask == 1))

    def test_refresh_tiles_cover_640_square_with_center_patch(self) -> None:
        tiles = detector.refresh_tiles((640, 640), 320)
        self.assertEqual(
            [(tile.x1, tile.y1, tile.x2, tile.y2, tile.reason) for tile in tiles],
            [
                (0, 0, 320, 640, "tile"),
                (320, 0, 640, 640, "tile"),
                (160, 0, 480, 640, "tile"),
            ],
        )
        self.assert_first_two_tiles_cover_frame(tiles, (640, 640))

    def test_refresh_tiles_cover_wide_frame_with_center_patch(self) -> None:
        tiles = detector.refresh_tiles((480, 640), 320)
        self.assertEqual(
            [(tile.x1, tile.y1, tile.x2, tile.y2, tile.reason) for tile in tiles],
            [
                (0, 0, 320, 480, "tile"),
                (320, 0, 640, 480, "tile"),
                (160, 0, 480, 480, "tile"),
            ],
        )
        self.assert_first_two_tiles_cover_frame(tiles, (480, 640))

    def test_refresh_tiles_cover_odd_width_frame_without_gap(self) -> None:
        tiles = detector.refresh_tiles((481, 641), 320)
        self.assertEqual(
            [(tile.x1, tile.y1, tile.x2, tile.y2, tile.reason) for tile in tiles],
            [
                (0, 0, 320, 481, "tile"),
                (320, 0, 641, 481, "tile"),
                (160, 0, 481, 481, "tile"),
            ],
        )
        self.assert_first_two_tiles_cover_frame(tiles, (481, 641))
        widths = [tile.width for tile in tiles[:2]]
        self.assertLessEqual(max(widths) - min(widths), 1)

    def test_tile_scheduler_spreads_one_cycle_across_tiles(self) -> None:
        scheduler = detector.RefreshTileScheduler("tiles", interval_ms=2000, tile_size=320)
        self.assertIsNone(scheduler.next_due_roi((640, 640), 10.0))
        self.assertEqual(scheduler.next_due_roi((640, 640), 12.0), detector.Roi(0, 0, 320, 640, "tile"))
        self.assertIsNone(scheduler.next_due_roi((640, 640), 12.5))
        self.assertEqual(scheduler.next_due_roi((640, 640), 12.667), detector.Roi(320, 0, 640, 640, "tile"))

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

    def test_non_full_roi_expands_to_include_historical_box_center(self) -> None:
        detections = np.array(
            [
                [40, 20, 90, 100, 0.9, 0],
                [420, 420, 460, 460, 0.8, 1],
            ],
            dtype=np.float32,
        )
        roi = detector.Roi(60, 40, 120, 120, "motion")

        expanded = detector.union_roi_with_detections(roi, detections, (480, 640), proximity_pixels=0)

        self.assertEqual(expanded, detector.Roi(40, 20, 120, 120, "motion"))

    def test_region_update_does_not_replace_global_cache(self) -> None:
        cache = detector.DetectionCache(ttl_frames=0)
        cache.update_full(
            np.array(
                [
                    [20, 20, 60, 60, 0.9, 0],
                    [420, 420, 460, 460, 0.8, 1],
                ],
                dtype=np.float32,
            ),
            frame_id=1,
        )
        cache.update_region(
            np.array([[25, 25, 65, 65, 0.95, 0]], dtype=np.float32),
            frame_id=2,
            roi=detector.Roi(0, 0, 120, 120, "tile"),
        )

        detections, cached_flags = cache.snapshot(frame_id=2)
        self.assertEqual(detections.shape[0], 2)
        self.assertEqual(cached_flags.count(False), 1)
        self.assertEqual(cached_flags.count(True), 1)
        self.assertTrue(np.any(detections[:, 0] == 420))

    def test_full_update_replaces_global_cache(self) -> None:
        cache = detector.DetectionCache(ttl_frames=0)
        cache.update_full(
            np.array(
                [
                    [20, 20, 60, 60, 0.9, 0],
                    [420, 420, 460, 460, 0.8, 1],
                ],
                dtype=np.float32,
            ),
            frame_id=1,
        )
        cache.update_full(np.array([[25, 25, 65, 65, 0.95, 0]], dtype=np.float32), frame_id=2)

        detections, cached_flags = cache.snapshot(frame_id=2)
        self.assertEqual(detections.shape[0], 1)
        self.assertEqual(cached_flags, [False])
        self.assertEqual(int(detections[0, 0]), 25)

    def test_ttl_prune_returns_expired_tracks_for_overlay(self) -> None:
        cache = detector.DetectionCache(ttl_frames=2)
        cache.update_full(np.array([[20, 20, 60, 60, 0.9, 0]], dtype=np.float32), frame_id=1)

        expired = cache.prune(frame_id=4)

        self.assertEqual(len(expired), 1)
        self.assertEqual(len(cache.pop_expired_tracks()), 1)
        detections, cached_flags = cache.snapshot(frame_id=4)
        self.assertEqual(detections.shape[0], 0)
        self.assertEqual(cached_flags, [])

    def test_expired_overlay_hides_after_display_window(self) -> None:
        overlay = detector.ExpiredDetectionOverlay(display_frames=2)
        track = detector.TrackedDetection(
            np.array([20, 20, 60, 60, 0.9, 0], dtype=np.float32),
            last_seen_frame=1,
        )

        overlay.add_tracks([track], frame_id=5)

        self.assertEqual(overlay.snapshot(frame_id=5).shape[0], 1)
        self.assertEqual(overlay.snapshot(frame_id=7).shape[0], 1)
        self.assertEqual(overlay.snapshot(frame_id=8).shape[0], 0)

    def test_frame_diff_generates_roi_for_small_motion(self) -> None:
        selector = detector.MotionRoiSelector(
            mode="roi",
            min_area=10_000,
            padding_pixels=0,
            frame_diff_enabled=True,
            frame_diff_threshold=12,
            frame_diff_min_area=20,
            frame_diff_alpha=0.0,
            smooth_alpha=0.0,
        )
        first = np.zeros((80, 80, 3), dtype=np.uint8)
        second = first.copy()
        second[30:40, 30:40] = 255

        self.assertIsNone(selector._motion_roi(first))
        roi = selector._motion_roi(second)

        self.assertIsNotNone(roi)
        assert roi is not None
        self.assertEqual(roi.reason, "motion")
        self.assertLessEqual(roi.x1, 30)
        self.assertLessEqual(roi.y1, 30)
        self.assertGreaterEqual(roi.x2, 40)
        self.assertGreaterEqual(roi.y2, 40)

    def test_frame_diff_respects_min_area(self) -> None:
        selector = detector.MotionRoiSelector(
            mode="roi",
            min_area=10_000,
            padding_pixels=0,
            frame_diff_enabled=True,
            frame_diff_threshold=12,
            frame_diff_min_area=200,
            frame_diff_alpha=0.0,
            smooth_alpha=0.0,
        )
        first = np.zeros((80, 80, 3), dtype=np.uint8)
        second = first.copy()
        second[30:35, 30:35] = 255

        self.assertIsNone(selector._motion_roi(first))
        self.assertIsNone(selector._motion_roi(second))

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
