from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import socket
import sys
import time
import types
import unittest
import urllib.request

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
        scheduler = detector.RefreshTileScheduler("tiles", interval_ms=1000, tile_size=320)
        self.assertIsNone(scheduler.next_due_roi((640, 640), 10.0))
        self.assertEqual(scheduler.next_due_roi((640, 640), 11.0), detector.Roi(0, 0, 320, 640, "tile"))
        self.assertIsNone(scheduler.next_due_roi((640, 640), 11.2))
        self.assertEqual(scheduler.next_due_roi((640, 640), 11.334), detector.Roi(320, 0, 640, 640, "tile"))

    def test_tile_scheduler_zero_interval_disables_refresh(self) -> None:
        scheduler = detector.RefreshTileScheduler("tiles", interval_ms=0, tile_size=320)
        self.assertIsNone(scheduler.next_due_roi((640, 640), 10.0))
        self.assertIsNone(scheduler.next_due_roi((640, 640), 30.0))

    def test_default_args_use_full_stream_detection_and_disable_ttl(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = ["detect_ncnn_yolov5.py", "--self-test"]
            args = detector.parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.motion_source, "mog2")
        self.assertEqual(args.threads, 2)
        self.assertEqual(args.roi_mode, "full")
        self.assertEqual(args.img_size, 320)
        self.assertEqual(args.input_name, "in0")
        self.assertEqual(args.output_name, "out0")
        self.assertEqual(args.full_frame_refresh_ms, 0.0)
        self.assertEqual(args.cached_roi_ms, 0.0)
        self.assertEqual(args.detection_ttl_ms, 0.0)
        self.assertEqual(args.stream_url, "http://127.0.0.1:8080/?action=stream")
        self.assertEqual(args.backend_url, "http://172.20.10.3:5000/api/detections")
        self.assertTrue(args.latest_json)
        self.assertEqual(args.latest_json_host, "0.0.0.0")
        self.assertEqual(args.latest_json_port, 8090)
        self.assertEqual(args.device_id, "pi-camera-01")
        self.assertEqual(detector.ttl_frames_from_args(args, 30.0), 0)

    def test_default_stream_url_does_not_conflict_with_other_input_modes(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = ["detect_ncnn_yolov5.py", "--image", "sample.jpg"]
            args = detector.parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.stream_url, "http://127.0.0.1:8080/?action=stream")
        self.assertEqual(args.image.name, "sample.jpg")
        self.assertFalse(detector.cli_option_provided(["--image", "sample.jpg"], "--stream-url"))

    def test_stream_url_arg_preserves_mjpg_streamer_url(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = [
                "detect_ncnn_yolov5.py",
                "--stream-url",
                "http://172.20.10.2:8080/?action=stream",
            ]
            args = detector.parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.stream_url, "http://172.20.10.2:8080/?action=stream")
        self.assertIsNone(args.video)
        self.assertIsNone(args.image)

    def test_stream_url_is_mutually_exclusive_with_other_input_modes(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = [
                "detect_ncnn_yolov5.py",
                "--stream-url",
                "http://172.20.10.2:8080/?action=stream",
                "--video",
                "sample.mp4",
            ]
            with self.assertRaises(SystemExit):
                detector.main()
        finally:
            sys.argv = original_argv

    def test_removed_overlapping_args_are_rejected(self) -> None:
        removed_args = [
            "--camera-fps",
            "--face-refresh-ms",
            "--full-frame-interval",
            "--cached-roi-interval",
            "--raw-stream",
            "--raw-stream-host",
            "--raw-stream-port",
            "--raw-stream-fps",
            "--raw-stream-width",
            "--raw-stream-quality",
            "--camera",
            "--frame-diff-enabled",
            "--no-frame-diff-enabled",
            "--detection-ttl-frames",
            "--stream-read-fps",
            "--profile-every",
        ]
        original_argv = sys.argv
        try:
            for arg in removed_args:
                with self.subTest(arg=arg):
                    sys.argv = ["detect_ncnn_yolov5.py", arg, "1"]
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        detector.parse_args()
        finally:
            sys.argv = original_argv

    def test_ttl_ms_uses_runtime_fps_unless_frames_are_explicit(self) -> None:
        args = argparse.Namespace(detection_ttl_ms=500.0)
        self.assertEqual(detector.ttl_frames_from_args(args, 30.0), 15)

        args.detection_ttl_ms = 0.0
        self.assertEqual(detector.ttl_frames_from_args(args, 30.0), 0)

    def test_cached_roi_due_respects_ms_interval(self) -> None:
        detections = np.array([[20, 20, 60, 60, 0.9, 0]], dtype=np.float32)

        self.assertFalse(detector.cached_roi_due(detections, now=10.0, last_refresh_time=9.8, interval_ms=500))
        self.assertTrue(detector.cached_roi_due(detections, now=10.31, last_refresh_time=9.8, interval_ms=500))
        self.assertFalse(detector.cached_roi_due(detections, now=10.31, last_refresh_time=9.8, interval_ms=0))
        self.assertFalse(
            detector.cached_roi_due(np.empty((0, 6), dtype=np.float32), now=10.31, last_refresh_time=9.8, interval_ms=500)
        )

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
            motion_source="frame_diff",
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
            motion_source="frame_diff",
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

    def test_mog2_source_ignores_frame_diff(self) -> None:
        class FakeSubtractor:
            def apply(self, _frame: np.ndarray) -> np.ndarray:
                return np.zeros((80, 80), dtype=np.uint8)

        selector = detector.MotionRoiSelector(
            mode="roi",
            min_area=20,
            padding_pixels=0,
            motion_source="mog2",
            frame_diff_threshold=12,
            frame_diff_min_area=20,
            frame_diff_alpha=0.0,
            smooth_alpha=0.0,
        )
        selector._subtractor = FakeSubtractor()

        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        changed = frame.copy()
        changed[30:50, 30:50] = 255
        self.assertIsNone(selector._motion_roi(frame))
        self.assertIsNone(selector._motion_roi(changed))

    def test_both_motion_source_merges_mog2_and_frame_diff(self) -> None:
        class FakeSubtractor:
            def apply(self, _frame: np.ndarray) -> np.ndarray:
                mask = np.zeros((80, 80), dtype=np.uint8)
                mask[5:18, 5:18] = 255
                return mask

        selector = detector.MotionRoiSelector(
            mode="roi",
            min_area=20,
            padding_pixels=0,
            motion_source="both",
            frame_diff_threshold=12,
            frame_diff_min_area=20,
            frame_diff_alpha=0.0,
            smooth_alpha=0.0,
        )
        selector._subtractor = FakeSubtractor()
        first = np.zeros((80, 80, 3), dtype=np.uint8)
        second = first.copy()
        second[50:62, 50:62] = 255

        selector._motion_roi(first)
        roi = selector._motion_roi(second)

        self.assertIsNotNone(roi)
        assert roi is not None
        self.assertLessEqual(roi.x1, 5)
        self.assertLessEqual(roi.y1, 5)
        self.assertGreaterEqual(roi.x2, 62)
        self.assertGreaterEqual(roi.y2, 62)

    def test_invalid_motion_source_raises(self) -> None:
        with self.assertRaises(ValueError):
            detector.MotionRoiSelector(motion_source="invalid")

    def test_latest_json_hub_returns_latest_payload(self) -> None:
        hub = detector.LatestJsonHub("pi-camera-01")
        initial = json.loads(hub.snapshot_bytes().decode("utf-8"))
        self.assertEqual(initial["status"], "warming_up")
        self.assertEqual(initial["device_id"], "pi-camera-01")

        hub.update({"device_id": "pi-camera-01", "frame_id": 7, "detections": [{"class_id": 1}]})
        latest = json.loads(hub.snapshot_bytes().decode("utf-8"))

        self.assertEqual(latest["frame_id"], 7)
        self.assertEqual(latest["detections"][0]["class_id"], 1)

    def test_latest_json_server_serves_payload_and_health(self) -> None:
        hub = detector.LatestJsonHub("pi-camera-01")
        hub.update({"device_id": "pi-camera-01", "frame_id": 9, "detections": []})
        server = detector.LatestJsonServer("127.0.0.1", 0, hub)
        server.start()
        try:
            with urllib.request.urlopen(server.url, timeout=2.0) as response:
                body = json.loads(response.read().decode("utf-8"))
                headers = response.headers
            with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/health", timeout=2.0) as response:
                health = json.loads(response.read().decode("utf-8"))
        finally:
            server.close()

        self.assertEqual(body["frame_id"], 9)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "*")

    def test_websocket_text_frame_encodes_small_payload(self) -> None:
        self.assertEqual(detector.websocket_text_frame(b"abc"), b"\x81\x03abc")

    def test_latest_json_server_websocket_sends_initial_payload(self) -> None:
        hub = detector.LatestJsonHub("pi-camera-01")
        hub.update({"device_id": "pi-camera-01", "frame_id": 11, "detections": []})
        server = detector.LatestJsonServer("127.0.0.1", 0, hub)
        server.start()
        key = base64_key = os.urandom(16)
        encoded_key = __import__("base64").b64encode(base64_key).decode("ascii")
        try:
            sock = socket.create_connection(("127.0.0.1", server.port), timeout=2.0)
            request = (
                "GET /ws HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{server.port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {encoded_key}\r\n"
                "Sec-WebSocket-Version: 13\r\n"
                "\r\n"
            ).encode("ascii")
            sock.sendall(request)
            response = sock.recv(4096)
            self.assertIn(b"101 Switching Protocols", response)
            header = sock.recv(2)
            self.assertEqual(header[0], 0x81)
            length = header[1] & 0x7F
            payload = sock.recv(length)
            body = json.loads(payload.decode("utf-8"))
        finally:
            try:
                sock.close()
            except Exception:
                pass
            server.close()

        self.assertEqual(body["frame_id"], 11)

    def test_latest_frame_stream_drops_stale_frames(self) -> None:
        class FakeCapture:
            def __init__(self) -> None:
                self.frames = [
                    np.full((2, 2, 3), value, dtype=np.uint8)
                    for value in (1, 2, 3)
                ]
                self.released = False

            def set(self, _prop: int, _value: float) -> None:
                return None

            def get(self, _prop: int) -> float:
                return 25.0

            def isOpened(self) -> bool:
                return True

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self.frames:
                    return True, self.frames.pop(0)
                time.sleep(0.01)
                return False, None

            def release(self) -> None:
                self.released = True

        capture = FakeCapture()
        stream = detector.LatestFrameStream("fake://stream", read_timeout=0.5, capture=capture)
        self.assertTrue(stream.is_opened())
        stream.start()
        assert stream._thread is not None
        stream._thread.join(timeout=1.0)

        ok, frame = stream.read()

        stream.release()
        self.assertTrue(ok)
        assert frame is not None
        self.assertEqual(int(frame[0, 0, 0]), 3)
        self.assertTrue(capture.released)

        ok, frame = stream.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)

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
