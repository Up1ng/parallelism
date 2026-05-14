import argparse
import logging
import multiprocessing
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

from classes import SensorX, SensorCam, WindowImage

LOG_DIR = Path(__file__).resolve().parents[1] / "log"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "task4.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
)


def parse_resolution(value: str) -> tuple:
    try:
        w, h = value.lower().split("x", 1)
        width, height = int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("Resolution must be WxH, e.g. 640x480")
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive")
    return width, height


def push_latest(q, value):
    try:
        q.put_nowait(value)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put(value, block=True, timeout=0.1)
        except queue.Full:
            pass


def sensor_worker(sensor, out_q, stop):
    while not stop.is_set():
        try:
            push_latest(out_q, sensor.get())
        except Exception:
            logging.exception("Sensor worker error")
            stop.set()


def drain_queue(q, fallback):
    value = fallback
    while True:
        try:
            value = q.get_nowait()
        except queue.Empty:
            return value


FREQ_LABELS = ["100 Hz", "10 Hz", "1 Hz"]


def build_frame(cam_frame, sensor_values: list, mode: str):
    img = cam_frame.copy() if cam_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    panel_h = 20 + len(sensor_values) * 28 + 10
    panel_w = 220
    x0 = img.shape[1] - panel_w - 10
    y0 = img.shape[0] - panel_h - 10

    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (245, 245, 245), -1)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (60, 60, 60), 1)

    for i, val in enumerate(sensor_values):
        text = f"S{i + 1} ({FREQ_LABELS[i]}): {val if val is not None else '---'}"
        cv2.putText(img, text, (x0 + 8, y0 + 22 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    label = f"Mode: {mode}"
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 200, 0) if mode == "threads" else (0, 100, 255), 1, cv2.LINE_AA)
    return img


def build_workers(sensors, queues, stop, mode):
    workers = []
    for i, (sensor, q) in enumerate(zip(sensors, queues)):
        use_process = (mode == "processes" and i > 0)
        if use_process:
            w = multiprocessing.Process(
                target=sensor_worker,
                args=(sensor, q, stop),
                name=f"sensor-proc-{i}",
                daemon=True,
            )
        else:
            w = threading.Thread(
                target=sensor_worker,
                args=(sensor, q, stop),
                name=f"sensor-thread-{i}",
                daemon=True,
            )
        workers.append(w)
    return workers


def main():
    parser = argparse.ArgumentParser(description="Task 4: webcam + SensorX with threads or processes")
    parser.add_argument("--camera", default="0", help="Camera device index or path (default: 0)")
    parser.add_argument("--resolution", type=parse_resolution, default=parse_resolution("640x480"),
                        help="Camera resolution WxH (default: 640x480)")
    parser.add_argument("--fps", type=float, default=30.0, help="Display FPS (default: 30)")
    parser.add_argument("--mode", choices=["threads", "processes"], default="threads",
                        help="Concurrency mode (default: threads)")
    args = parser.parse_args()

    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    logging.info("Starting in '%s' mode", args.mode)

    if args.mode == "processes":
        stop = multiprocessing.Event()
        queues = [queue.Queue(maxsize=1)] + [multiprocessing.Queue(maxsize=1) for _ in range(3)]
    else:
        stop = threading.Event()
        queues = [queue.Queue(maxsize=1) for _ in range(4)]

    sensors = [
        SensorCam(args.camera, args.resolution),
        SensorX(0.01),
        SensorX(0.1),
        SensorX(1.0),
    ]

    workers = build_workers(sensors, queues, stop, args.mode)
    for w in workers:
        w.start()

    window = WindowImage(args.fps)
    latest = [None] * 4

    try:
        while not stop.is_set():
            for i, q in enumerate(queues):
                latest[i] = drain_queue(q, latest[i])

            frame = build_frame(latest[0], latest[1:], args.mode)

            if window.show(frame) == ord("q"):
                stop.set()
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=2.0)
        del window
        for s in sensors:
            del s


if __name__ == "__main__":
    main()
