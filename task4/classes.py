import logging
import time

import cv2


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement get()")


class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._counter = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._counter += 1
        return self._counter


class SensorCam(Sensor):
    def __init__(self, cam_id: str, resolution: tuple):
        source = int(cam_id) if cam_id.isdigit() else cam_id
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            logging.error("Cannot open camera: %s", cam_id)
            raise RuntimeError(f"Cannot open camera: {cam_id}")
        w, h = resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        logging.info("Camera '%s' opened at %dx%d", cam_id, w, h)

    def get(self):
        ret, frame = self._cap.read()
        if not ret or frame is None:
            logging.error("Failed to capture frame from camera")
            raise RuntimeError("Camera read failed")
        return frame

    def __del__(self):
        if hasattr(self, "_cap"):
            self._cap.release()


class WindowImage:
    def __init__(self, fps: float, title: str = "Task 4"):
        self._title = title
        self._wait_ms = max(1, int(1000 / fps))
        try:
            cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        except cv2.error:
            logging.exception("Failed to create window")
            raise RuntimeError("Window creation failed")
        logging.info("Window '%s' ready, delay=%d ms", self._title, self._wait_ms)

    def show(self, img) -> int:
        try:
            cv2.imshow(self._title, img)
            return cv2.waitKey(self._wait_ms) & 0xFF
        except cv2.error:
            logging.exception("Failed to display frame")
            raise RuntimeError("Window display failed")

    def __del__(self):
        try:
            cv2.destroyWindow(self._title)
        except cv2.error:
            pass
