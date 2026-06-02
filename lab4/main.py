#!/usr/bin/env python3
"""
Launch:
  python main.py --camera 0 --resolution 1280x720 --fps 30 --mode thread
  python main.py --mode process
  python main.py --mode benchmark
  q - exit
"""

import argparse
import logging
import multiprocessing
import queue
import statistics
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# Logging
_LOG_DIR = Path(__file__).parent / "log"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Sensor (basic class)
class Sensor(ABC):
    @abstractmethod
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


# SensorX
class SensorX(Sensor):
    """Sensor X"""

    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


# SensorCam
class SensorCam(Sensor):
    """camera. RAII: opens in __init__, releases in __del__."""
    def __init__(self, camera_name: str, resolution: tuple[int, int]):
        try:
            cam_id = int(camera_name)
        except ValueError:
            cam_id = camera_name

        self._cam = cv2.VideoCapture(cam_id)
        if not self._cam.isOpened():
            msg = f"Cannot open camera '{camera_name}'"
            logger.error(msg)
            raise RuntimeError(msg)

        w, h = resolution
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        logger.info(f"Camera '{camera_name}' opened at {w}x{h}")

    def get(self) -> Optional[np.ndarray]:
        ret, frame = self._cam.read()
        if not ret:
            logger.error("Failed to read frame — camera may have been disconnected")
            return None
        return frame

    def __del__(self):
        if hasattr(self, "_cam") and self._cam is not None:
            self._cam.release()
            logger.info("Camera released")


# WindowImage
class WindowImage:
    """OpenCV window. RAII: created in __init__, destroyed in __del__."""

    _NAME = "Sensor Dashboard"

    def __init__(self, fps: float):
        self._wait_ms = max(1, int(1000.0 / fps))
        try:
            cv2.namedWindow(self._NAME, cv2.WINDOW_AUTOSIZE)
        except Exception as exc:
            logger.error(f"Cannot create window: {exc}")
            raise

    def show(self, img: np.ndarray) -> bool:
        """Shows a frame. Returns False when 'q' is pressed."""
        try:
            cv2.imshow(self._NAME, img)
            return (cv2.waitKey(self._wait_ms) & 0xFF) != ord("q")
        except Exception as exc:
            logger.error(f"Display error: {exc}")
            return False

    def __del__(self):
        try:
            cv2.destroyWindow(self._NAME)
        except Exception:
            pass


# Helpers
def put_latest(q, item) -> None:
    """Puts only the latest value: discards the old one if the queue is full."""
    try:
        q.put_nowait(item)
    except Exception:
        try:
            q.get_nowait()
        except Exception:
            pass
        try:
            q.put_nowait(item)
        except Exception:
            pass


def build_frame(frame: Optional[np.ndarray], values: dict) -> np.ndarray:
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = frame.copy()
    h, w = out.shape[:2]
    start_y = h - 15 - 22 * len(values)
    for i, (name, val) in enumerate(values.items()):
        cv2.putText(out, f"{name}: {val}", (w - 200, start_y + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# Worker functions
def thread_worker(sensor: Sensor, q: queue.Queue, stop: threading.Event) -> None:
    while not stop.is_set():
        data = sensor.get()
        if data is not None:
            put_latest(q, (data, time.time()))
        else:
            stop.set()


def process_worker(delay: float, q, stop) -> None:
    """Runs SensorX inside a child process."""
    sensor = SensorX(delay)
    while not stop.is_set():
        data = sensor.get()
        put_latest(q, (data, time.time()))


# Thread mode
def run_thread_mode(camera_name: str, resolution: tuple[int, int], fps: float) -> dict:
    sensors = {"Sensor0": SensorX(0.01), "Sensor1": SensorX(0.10), "Sensor2": SensorX(1.00)}
    queues = {name: queue.Queue(maxsize=1) for name in sensors}
    stop = threading.Event()
    threads = []

    for name, sensor in sensors.items():
        t = threading.Thread(target=thread_worker, args=(sensor, queues[name], stop), daemon=True)
        t.start()
        threads.append(t)

    cam_q: queue.Queue = queue.Queue(maxsize=1)
    cam_sensor = SensorCam(camera_name, resolution)
    t = threading.Thread(target=thread_worker, args=(cam_sensor, cam_q, stop), daemon=True)
    t.start()
    threads.append(t)

    window = WindowImage(fps)
    values = {name: 0 for name in sensors}
    frame = None
    latencies = {name: [] for name in sensors}

    try:
        running = True
        while running and not stop.is_set():
            for name in sensors:
                try:
                    val, ts = queues[name].get_nowait()
                    values[name] = val
                    latencies[name].append(time.time() - ts)
                except queue.Empty:
                    pass
            try:
                frame, _ = cam_q.get_nowait()
            except queue.Empty:
                pass
            running = window.show(build_frame(frame, values))
    except KeyboardInterrupt:
        pass

    stop.set()
    for t in threads:
        t.join(timeout=2.0)
    return latencies


# Process mode
def run_process_mode(camera_name: str, resolution: tuple[int, int], fps: float) -> dict:
    """
    SensorX — in processes, SensorCam — in a thread.
    On macOS multiprocessing uses 'spawn', so:
      - worker functions must be at the module level
      - cv2.VideoCapture cannot be passed to a process (not picklable)
    """
    delays = {"Sensor0": 0.01, "Sensor1": 0.10, "Sensor2": 1.00}
    mp_qs = {name: multiprocessing.Queue(maxsize=1) for name in delays}
    mp_stop = multiprocessing.Event()
    processes = []

    for name, delay in delays.items():
        p = multiprocessing.Process(target=process_worker, args=(delay, mp_qs[name], mp_stop), daemon=True)
        p.start()
        processes.append(p)

    thr_stop = threading.Event()
    cam_q: queue.Queue = queue.Queue(maxsize=1)
    cam_sensor = SensorCam(camera_name, resolution)
    cam_thread = threading.Thread(target=thread_worker, args=(cam_sensor, cam_q, thr_stop), daemon=True)
    cam_thread.start()

    window = WindowImage(fps)
    values = {name: 0 for name in delays}
    frame = None
    latencies = {name: [] for name in delays}

    try:
        running = True
        while running and not thr_stop.is_set():
            for name in delays:
                try:
                    val, ts = mp_qs[name].get_nowait()
                    values[name] = val
                    latencies[name].append(time.time() - ts)
                except queue.Empty:
                    pass
            try:
                frame, _ = cam_q.get_nowait()
            except queue.Empty:
                pass
            running = window.show(build_frame(frame, values))
    except KeyboardInterrupt:
        pass

    mp_stop.set()
    thr_stop.set()
    for p in processes:
        p.join(timeout=2.0)
    if cam_thread:
        cam_thread.join(timeout=2.0)
    return latencies


# Benchmark
def benchmark(duration: float = 5.0) -> None:
    """Compares IPC latency: thread vs process, without display."""
    print(f"\n{'─'*55}")
    print(f"  Benchmark: thread vs process  ({duration}s per mode)")
    print(f"  Sensor: SensorX(0.01) — nominal 10 ms / cycle")
    print(f"{'─'*55}")

    def collect(q, duration_s: float) -> list[float]:
        lats, end = [], time.time() + duration_s
        while time.time() < end:
            try:
                _, ts = q.get_nowait()
                lats.append((time.time() - ts) * 1000)
            except Exception:
                time.sleep(0.001)
        return lats

    # Thread
    q_t: queue.Queue = queue.Queue(maxsize=1)
    stop_t = threading.Event()
    thr = threading.Thread(target=thread_worker, args=(SensorX(0.01), q_t, stop_t), daemon=True)
    thr.start()
    thread_lats = collect(q_t, duration)
    stop_t.set()
    thr.join(timeout=2.0)

    # Process
    q_p = multiprocessing.Queue(maxsize=1)
    stop_p = multiprocessing.Event()
    proc = multiprocessing.Process(target=process_worker, args=(0.01, q_p, stop_p), daemon=True)
    proc.start()
    proc_lats = collect(q_p, duration)
    stop_p.set()
    proc.join(timeout=2.0)

    def report(label: str, lats: list[float]) -> None:
        if not lats:
            print(f"  {label}: no data"); return
        print(f"\n  {label}  ({len(lats)} samples)")
        print(f"    mean   {statistics.mean(lats):.3f} ms")
        print(f"    median {statistics.median(lats):.3f} ms")
        if len(lats) > 1:
            print(f"    stdev  {statistics.stdev(lats):.3f} ms")
        print(f"    min    {min(lats):.3f} ms  /  max  {max(lats):.3f} ms")

    report("Thread ", thread_lats)
    report("Process", proc_lats)

    if thread_lats and proc_lats:
        ratio = statistics.mean(proc_lats) / max(statistics.mean(thread_lats), 1e-9)
        print(f"\n  Process is {ratio:.1f}x slower than thread")
        print(f"  (IPC overhead: pickling + pipe transfer + unpickling)")
    print(f"{'─'*55}\n")


# CLI
def parse_resolution(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(f"Format: WxH (e.g. 1280x720), got: '{s}'")


def print_latency_summary(mode: str, latencies: dict) -> None:
    print(f"\n── {mode} — latency summary ──")
    for name, lats in latencies.items():
        if lats:
            mean_us = statistics.mean(lats) * 1_000_000
            p99_us = sorted(lats)[int(0.99 * len(lats))] * 1_000_000
            print(f"  {name}: mean {mean_us:6.1f} µs   p99 {p99_us:7.1f} µs   (n={len(lats)})")
        else:
            print(f"  {name}: no data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise 04 — Sensor Dashboard")
    parser.add_argument("--camera", default="0", help="Camera index or path (default: 0)")
    parser.add_argument("--resolution", default="1280x720", type=parse_resolution, metavar="WxH")
    parser.add_argument("--fps", type=float, default=30.0, help="Window refresh rate in Hz")
    parser.add_argument("--mode", choices=["thread", "process", "benchmark"], default="thread",
                        help="thread | process | benchmark")
    args = parser.parse_args()

    if args.mode == "benchmark":
        benchmark()
        return

    try:
        if args.mode == "thread":
            print("Mode: THREAD — all sensors in threads")
            lats = run_thread_mode(args.camera, args.resolution, args.fps)
        else:
            print("Mode: PROCESS — SensorX in processes, SensorCam in thread")
            lats = run_process_mode(args.camera, args.resolution, args.fps)
    except RuntimeError as exc:
        logger.error(f"Unrecoverable camera error: {exc}")
        sys.exit(1)

    print_latency_summary(args.mode.capitalize(), lats)


if __name__ == "__main__":
    main()