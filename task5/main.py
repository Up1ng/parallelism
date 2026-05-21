import argparse
import multiprocessing
import os
import queue
import threading
import time

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolov8s-pose.pt"
_SENTINEL = object()


class VideoCapture:

    def __init__(self, source):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        self.release()

    def release(self):
        if self._cap.isOpened():
            self._cap.release()

    def read(self):
        return self._cap.read()

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


class VideoWriter:

    def __init__(self, path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {path}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        self.release()

    def release(self):
        if self._writer.isOpened():
            self._writer.release()

    def write(self, frame):
        self._writer.write(frame)


def process_single_thread(video_path, output_path):

    model = YOLO(MODEL_PATH)

    with VideoCapture(video_path) as cap:
        fps, width, height = cap.fps, cap.width, cap.height
        frames_processed = 0

        with VideoWriter(output_path, fps, width, height) as writer:
            start = time.perf_counter()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = model(frame, verbose=False)
                writer.write(result[0].plot())
                frames_processed += 1
            elapsed = time.perf_counter() - start

    print(f"  {frames_processed} frames processed")
    return elapsed


def _inference_worker(in_q, out, ready):

    model = YOLO(MODEL_PATH)
    ready.wait()
    while True:
        item = in_q.get()
        if item is _SENTINEL:
            break
        idx, frame = item
        result = model(frame, verbose=False)
        out[idx] = result[0].plot()


def process_multi_thread(video_path, output_path, num_threads):

    in_q = queue.Queue(maxsize=num_threads * 2)
    out = {}
    ready = threading.Barrier(num_threads + 1)

    workers = [
        threading.Thread(target=_inference_worker, args=(in_q, out, ready), daemon=True)
        for _ in range(num_threads)
    ]
    for w in workers:
        w.start()

    with VideoCapture(video_path) as cap:
        fps, width, height = cap.fps, cap.width, cap.height
        frame_idx = 0

        with VideoWriter(output_path, fps, width, height) as writer:
            ready.wait()
            start = time.perf_counter()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                in_q.put((frame_idx, frame.copy()))
                frame_idx += 1

            for _ in workers:
                in_q.put(_SENTINEL)

            for w in workers:
                w.join()

            for i in range(frame_idx):
                writer.write(out[i])

            elapsed = time.perf_counter() - start

    print(f"  {frame_idx} frames processed with {num_threads} threads")
    return elapsed


def run_benchmark(video_path, output_prefix):
    """Test multiple thread counts and print a speedup table."""
    cpu_count = multiprocessing.cpu_count()
    counts = sorted({1, 2, 4, 8, cpu_count, cpu_count * 2})
    print(f"\n  CPU logical cores: {cpu_count}")
    print(f"  Testing thread counts: {counts}\n")

    results = []
    for n in counts:
        out_path = f"{output_prefix}_t{n}.mp4"
        if n == 1:
            t = process_single_thread(video_path, out_path)
        else:
            t = process_multi_thread(video_path, out_path, n)
        results.append((n, t))
        print(f"  {n:2d} thread(s): {t:.2f}s")

    baseline = results[0][1]
    print("\n  --- Speedup table ---")
    print(f"  {'Threads':>8} | {'Time (s)':>10} | {'Speedup':>8}")
    print("  " + "-" * 34)
    for n, t in results:
        print(f"  {n:>8} | {t:>10.2f} | {baseline / t:>8.2f}x")

    best_n, best_t = min(results, key=lambda x: x[1])
    print(f"\n  Optimal: {best_n} threads  ({baseline / best_t:.2f}x speedup)")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8s-pose CPU inference  (single / multi-thread / benchmark)"
    )
    parser.add_argument("--video", "-v", help="Path to input video (640x480)")
    parser.add_argument(
        "--mode", "-m",
        choices=["single", "multi", "benchmark"],
        default="single",
        help="Execution mode (default: single)",
    )
    parser.add_argument("--output", "-o", default="output.mp4", help="Output video filename")
    parser.add_argument(
        "--threads", "-t", type=int, default=4,
        help="Worker thread count for --mode multi (default: 4)",
    )
    args = parser.parse_args()

    if not args.video:
        parser.error("--video/-v is required for single / multi / benchmark modes")
    if not os.path.isfile(args.video):
        parser.error(f"Video file not found: {args.video}")

    if args.mode == "single":
        print(f"[Single-thread]  {args.video} -> {args.output}")
        elapsed = process_single_thread(args.video, args.output)
        print(f"Time: {elapsed:.3f}s")

    elif args.mode == "multi":
        print(f"[Multi-thread ({args.threads})]  {args.video} -> {args.output}")
        elapsed = process_multi_thread(args.video, args.output, args.threads)
        print(f"Time: {elapsed:.3f}s")

    elif args.mode == "benchmark":
        base = os.path.splitext(args.output)[0]
        run_benchmark(args.video, base)


if __name__ == "__main__":
    main()
