#!/usr/bin/env python3
"""
YOLOv8s-pose RealTime camera inference

Pipeline:
    MainThread   --(raw.copy())--> in_q (maxsize=N, drop when full)
    InferWorkers --> AnnotationStore (latest annotated frame)
    MainThread   --> shows ann if available, otherwise raw → imshow

Press 'q' to quit.

Usage:
    python realtime.py
    python realtime.py --threads 1   # MPS: 1 is usually optimal
    python realtime.py --device cpu --threads 4
"""

import argparse
import os
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# RAII wrappers

class CameraCapture:
    """RAII: открывает камеру в __init__, освобождает в __exit__."""

    def __init__(self, device: int = 0, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device: {device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._cap.release()

    def read(self):
        return self._cap.read()


class ModelInstance:
    """RAII: загружает YOLO на устройство в __init__, освобождает в __exit__."""

    def __init__(self, model_name: str, device: str):
        self._model  = YOLO(model_name)
        self._model.to(device)
        self._device = device

    def __enter__(self):
        return self

    def __exit__(self, *_):
        del self._model

    def infer(self, frame: np.ndarray) -> np.ndarray:
        return self._model(frame, verbose=False, device=self._device)[0].plot()


class AnnotationStore:
    """Thread-safe: хранит последний аннотированный кадр.
    Кадры с seq <= текущего игнорируются — только вперёд.
    """

    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None
        self._seq   = -1
        self._count = 0

    def put(self, seq: int, frame: np.ndarray) -> None:
        with self._lock:
            if seq > self._seq:
                self._frame  = frame
                self._seq    = seq
                self._count += 1

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def pop_count(self) -> int:
        with self._lock:
            n, self._count = self._count, 0
            return n


# Worker thread

def _infer_worker(model_name: str, device: str,
                  in_q: queue.Queue, store: AnnotationStore,
                  stop: threading.Event) -> None:
    timeout = 0.05  # начальная оценка до первого замера
    with ModelInstance(model_name, device) as model:
        while not stop.is_set():
            try:
                seq, frame = in_q.get(timeout=timeout)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            store.put(seq, model.infer(frame))
            timeout = (time.perf_counter() - t0) * 1.1  # inference + 10%


# Main realtime loop

def run_realtime(model_name: str, device: str,
                 num_threads: int, camera_device: int) -> None:
    in_q  = queue.Queue(maxsize=num_threads)
    store = AnnotationStore()
    stop  = threading.Event()

    workers = [
        threading.Thread(target=_infer_worker,
                         args=(model_name, device, in_q, store, stop),
                         daemon=True, name=f"infer-{i}")
        for i in range(num_threads)
    ]
    for w in workers:
        w.start()

    cap_times: deque = deque(maxlen=60)
    inf_t0  = time.perf_counter()
    inf_fps = 0.0
    seq     = 0

    print(f"RealTime | device={device} | threads={num_threads} | camera={camera_device}")
    print("Загрузка модели... Нажмите 'q' для выхода.")

    with CameraCapture(camera_device) as cam:
        while True:
            ok, raw = cam.read()   # единственное место чтения камеры
            if not ok:
                break

            # Отправляем на inference; если worker занят — дропаем, не ждём
            try:
                in_q.put_nowait((seq, raw.copy()))
            except queue.Full:
                pass
            seq += 1

            # FPS захвата
            now = time.perf_counter()
            cap_times.append(now)
            cap_fps = (
                (len(cap_times) - 1) / (cap_times[-1] - cap_times[0])
                if len(cap_times) > 1 else 0.0
            )

            # FPS inference (раз в секунду)
            elapsed = now - inf_t0
            if elapsed >= 1.0:
                inf_fps = store.pop_count() / elapsed
                inf_t0  = now

            # Показываем последний аннотированный кадр или raw пока inference не готов.
            # .copy() обязателен: cv2.putText мутирует массив in-place,
            # без копии HUD-текст накапливался бы на store._frame с каждым кадром.
            ann     = store.get()
            display = ann.copy() if ann is not None else raw

            # HUD
            cv2.putText(display, f"Camera: {cap_fps:.0f} fps",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,   0), 2)
            cv2.putText(display, f"Infer:  {inf_fps:.0f} fps  [{device}]",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow("YOLOv8s-pose RealTime", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    stop.set()
    cv2.destroyAllWindows()
    print("Остановлено.")


# Entry point

def main() -> None:
    detected = best_device()

    parser = argparse.ArgumentParser(
        description="YOLOv8s-pose RealTime",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--threads", "-t", type=int, default=None,
                        help="Число inference thread'ов. MPS: 1 оптимально, CPU: cpu_count")
    parser.add_argument("--device", default=detected,
                        choices=["mps", "cuda", "cpu"],
                        help=f"Устройство вычислений (определено: {detected})")
    parser.add_argument("--model", default="yolov8s-pose.pt",
                        help="Модель YOLO. yolov8n-pose.pt — быстрее")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    device = args.device

    if args.threads is not None:
        n = args.threads
    elif device in ("mps", "cuda"):
        n = 1       # GPU обрабатывает весь батч; больше thread'ов не помогут
    else:
        n = os.cpu_count() or 4

    run_realtime(args.model, device, n, args.camera)


if __name__ == "__main__":
    main()