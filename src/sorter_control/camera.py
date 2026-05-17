from __future__ import annotations

from typing import Any

from sorter_control.config import CameraConfig


class CameraCapture:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cap: Any | None = None

    def open(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for camera capture") from exc
        self._cap = cv2.VideoCapture(self.config.device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.exposure is not None:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
        if not self._cap.isOpened():
            raise RuntimeError(f"failed to open camera device {self.config.device_index}")

    def read(self) -> Any:
        if self._cap is None:
            self.open()
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("failed to read camera frame")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
