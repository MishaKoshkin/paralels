#!/usr/bin/env python3
"""
Usage:
    python inference.py input.mp4 --mode single   --output out.mp4
    python inference.py input.mp4 --mode multi    --threads 4 --output out.mp4
    python inference.py input.mp4 --mode benchmark
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# RAII wrappers

class VideoCapture:
    """RAII: открывает захват в __init__, освобождает в __exit__."""

    def __init__(self, source: str | int):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._cap.release()

    @property
    def fps(self) -> float:
        v = self._cap.get(cv2.CAP_PROP_FPS)
        return v if v > 0 else 30.0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 0:
            return n
        # Запасной вариант для контейнеров без метаданных о числе кадров
        pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        count = 0
        while self._cap.grab():
            count += 1
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        return count

    def read_all(self) -> list:
        frames = []
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames


class VideoWriter:
    """RAII: открывает запись в __init__, освобождает в __exit__."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._w = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._w.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter: {path}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._w.release()

    def write(self, frame) -> None:
        self._w.write(frame)


class ModelInstance:
    """RAII: загружает YOLO на устройство в __init__, удаляет в __exit__.
    Каждый поток должен иметь свой экземпляр (thread-safe паттерн inference).
    """

    def __init__(self, model_name: str = "yolov8s-pose.pt", device: str = "cpu"):
        self._model = YOLO(model_name)
        self._model.to(device)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def infer(self, frame):
        res = self._model(frame, verbose=False)
        return res[0].plot()


# Worker

_STOP = object()  # poison-pill sentinel


def _worker(model_name: str, device: str,
            in_q: queue.Queue, out_q: queue.Queue,
            errors: list) -> None:
    """Worker thread: создаёт свой ModelInstance, вычитывает in_q до sentinel."""
    try:
        model_instance = ModelInstance(model_name, device)
    except Exception as exc:
        errors.append((-1, exc))
        # Вычитываем queue чтобы in_q.join() не завис без этого worker'а
        while True:
            item = in_q.get()
            in_q.task_done()
            if item is _STOP:
                return

    with model_instance as model:
        while True:
            item = in_q.get()
            if item is _STOP:
                in_q.task_done()
                return
            idx, frame = item
            try:
                annotated = model.infer(frame)
                out_q.put((idx, annotated))
            except Exception as exc:
                errors.append((idx, exc))
            finally:
                in_q.task_done()


# Single-threaded

def process_single(video_path: str, output_path: str, model_name: str,
                   device: str = "cpu") -> float:
    """Читает все кадры, выполняет inference последовательно, записывает результат. Возвращает время inference (с)."""
    with VideoCapture(video_path) as cap:
        fps, w, h = cap.fps, cap.width, cap.height
        frames = cap.read_all()

    n = len(frames)
    print(f"[single]  {n} кадров | {fps:.1f} fps | {w}×{h} | device={device}")

    t0 = time.perf_counter()
    with ModelInstance(model_name, device) as model:
        annotated = [model.infer(f) for f in frames]
    elapsed = time.perf_counter() - t0

    with VideoWriter(output_path, fps, w, h) as writer:
        for f in annotated:
            writer.write(f)

    print(f"[single]  {elapsed:.3f} с  |  {n / elapsed:.2f} fps пропускная способность")
    return elapsed


# Multi-threaded

def process_multi(
    video_path: str,
    output_path: str,
    model_name: str,
    num_threads: int,
    device: str = "cpu",
) -> float:
    """
    Producer-consumer pipeline с упорядочиванием кадров по индексу.
    Возвращает время inference (с).
    """
    with VideoCapture(video_path) as cap:
        fps, w, h = cap.fps, cap.width, cap.height
        frames = cap.read_all()

    n = len(frames)
    print(f"[multi/{num_threads}]  {n} кадров | {fps:.1f} fps | {w}×{h} | device={device}")

    in_q: queue.Queue = queue.Queue(maxsize=num_threads * 8)
    out_q: queue.Queue = queue.Queue()
    errors: list = []

    workers = [
        threading.Thread(
            target=_worker,
            args=(model_name, device, in_q, out_q, errors),
            daemon=True,
            name=f"yolo-worker-{i}",
        )
        for i in range(num_threads)
    ]
    for wt in workers:
        wt.start()

    t0 = time.perf_counter()

    # Кладём кадры во входную queue
    for idx, frame in enumerate(frames):
        in_q.put((idx, frame))
    # Отправляем по одному poison pill на каждый worker
    for _ in workers:
        in_q.put(_STOP)

    in_q.join()  # ждём пока все task_done() не будут вызваны
    for wt in workers:
        wt.join(timeout=30)
        if wt.is_alive():
            raise RuntimeError(f"{wt.name} did not finish within timeout")
    elapsed = time.perf_counter() - t0

    if errors:
        idx, exc = errors[0]
        if idx == -1:
            raise RuntimeError("Worker failed to initialize model") from exc
        raise RuntimeError(f"Worker failed on frame {idx}") from exc

    # Собираем результаты и восстанавливаем исходный порядок
    result_map: dict[int, object] = {}
    while not out_q.empty():
        idx, annotated = out_q.get_nowait()
        result_map[idx] = annotated

    if len(result_map) != n:
        missing = [i for i in range(n) if i not in result_map]
        raise RuntimeError(f"Lost {n - len(result_map)} frame(s): indices {missing[:10]}")

    with VideoWriter(output_path, fps, w, h) as writer:
        for i in range(n):
            writer.write(result_map[i])

    print(f"[multi/{num_threads}]  {elapsed:.3f} с  |  {n / elapsed:.2f} fps пропускная способность")
    return elapsed


# Benchmark

def benchmark(video_path: str, model_name: str, device: str = "cpu",
              runs: int = 20) -> None:
    """Запускает каждую конфигурацию (device, threads) `runs` раз, выводит таблицу speedup.

    device="all" тестирует CPU (разное число потоков) + все доступные GPU,
    baseline всегда cpu/1 чтобы speedup'ы были напрямую сравнимы.
    """
    import statistics

    cpu_count = os.cpu_count() or 4

    if device == "all":
        candidates: list[tuple[str, int]] = [
            ("cpu", n) for n in sorted({1, 2, 4, max(1, cpu_count // 2), cpu_count})
        ]
        if torch.backends.mps.is_available():
            candidates.append(("mps", 1))
        if torch.cuda.is_available():
            candidates.append(("cuda", 1))
    elif device in ("mps", "cuda"):
        candidates = [(device, 1)]
    else:
        candidates = [(device, n)
                      for n in sorted({1, 2, 4, max(1, cpu_count // 2), cpu_count})]

    with VideoCapture(video_path) as cap:
        n_frames = cap.frame_count

    print(f"Benchmark: {candidates} × {runs} runs")
    print(f"Видео: {n_frames} кадров. Ожидаемое время: долго.\n")

    all_timings: dict[tuple[str, int], list[float]] = {}

    for dev, n in candidates:
        key = (dev, n)
        all_timings[key] = []
        for run in range(1, runs + 1):
            tmp = f"/tmp/_yolo_bench_{dev}_{n}.mp4"
            if n == 1:
                t = process_single(video_path, tmp, model_name, dev)
            else:
                t = process_multi(video_path, tmp, model_name, n, dev)
            all_timings[key].append(t)
            Path(tmp).unlink(missing_ok=True)
            print(f"  [{dev}/{n}t] run {run}/{runs}: {t:.3f}s")
        print()

    means = {k: statistics.mean(ts) for k, ts in all_timings.items()}
    stds  = {k: statistics.stdev(ts) if len(ts) > 1 else 0.0
             for k, ts in all_timings.items()}

    base_key = ("cpu", 1) if ("cpu", 1) in means else candidates[0]
    base = means[base_key]

    sep = "=" * 66
    print(sep)
    print(f"{'Device':>8}  {'Threads':>7}  {'Mean (s)':>10}  "
          f"{'±Std':>8}  {'Speedup':>8}  {'FPS':>8}")
    print("-" * 66)
    for dev, n in candidates:
        key = (dev, n)
        m, s = means[key], stds[key]
        print(f"{dev:>8}  {n:>7}  {m:>10.3f}  {s:>8.3f}  "
              f"{base/m:>8.2f}x  {n_frames/m:>8.2f}")
    print(sep)

    optimal = min(means, key=means.get)
    dev_opt, n_opt = optimal
    print(
        f"Optimal: {dev_opt} / {n_opt} thread(s)  "
        f"→  {base / means[optimal]:.2f}x speedup vs cpu/1  "
        f"(среднее за {runs} запусков)"
    )


# Entry point

def main() -> None:
    detected = best_device()

    parser = argparse.ArgumentParser(
        description="YOLOv8s-pose видео inference: однопоточный / многопоточный",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="Путь к входному видео (рекомендуется 640×480)")
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "benchmark"],
        default="multi",
        help="Режим выполнения",
    )
    parser.add_argument(
        "--output", "-o",
        default="output.mp4",
        help="Имя выходного видеофайла",
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=None,
        help="Число worker thread'ов для multi режима (по умолчанию: 1 для GPU, cpu_count для CPU)",
    )
    parser.add_argument(
        "--device",
        default=detected,
        choices=["auto", "all", "mps", "cuda", "cpu"],
        help=f"Устройство вычислений (определено: {detected})",
    )
    parser.add_argument(
        "--model",
        default="yolov8s-pose.pt",
        help="Имя модели YOLO или путь к локальному файлу",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=20,
        help="Number of runs to average in benchmark mode (default: 20)",
    )
    args = parser.parse_args()

    if not Path(args.video).exists():
        sys.exit(f"Ошибка: файл не найден: {args.video}")

    if args.device == "all" and args.mode != "benchmark":
        sys.exit("Ошибка: --device all поддерживается только с --mode benchmark")

    device = detected if args.device == "auto" else args.device

    if args.threads is not None:
        n = args.threads
    elif device in ("mps", "cuda"):
        n = 1
    else:
        n = os.cpu_count() or 4

    if args.mode == "benchmark":
        benchmark(args.video, args.model, device, args.runs)
    elif args.mode == "single":
        process_single(args.video, args.output, args.model, device)
    else:
        process_multi(args.video, args.output, args.model, n, device)


if __name__ == "__main__":
    main()