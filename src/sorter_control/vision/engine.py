from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sorter_control.config import AIConfig
from sorter_control.vision.classifier import VisionObservation


@dataclass
class ModelStatus:
    backend: str
    model_path: str
    loaded: bool
    message: str = ""


class VisionEngine:
    """ONNX/TensorRT-ready inference adapter with safe simulator fallback.

    The TensorRT path is intentionally isolated behind this adapter so the
    control loop can be tested on a laptop without Jetson libraries installed.
    """

    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self.session: Any | None = None
        self.status = self._load_backend()

    def _load_backend(self) -> ModelStatus:
        model_path = Path(self.config.model_path)
        if self.config.backend in ("auto", "onnx") and model_path.exists():
            try:
                import onnxruntime as ort  # type: ignore
            except ImportError:
                return ModelStatus("sim", str(model_path), False, "onnxruntime not installed")
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            return ModelStatus("onnx", str(model_path), True)
        return ModelStatus("sim", str(model_path), False, "model file missing; using simulator")

    def infer(self, frame: Any) -> VisionObservation:
        if self.session is None:
            # Deterministic fallback keeps the rest of the system debuggable.
            return VisionObservation(shape="cube", color="red", confidence=0.99, bbox=(0, 0, 40, 40))
        # Real YOLO postprocessing depends on the exported model head. Keep a
        # narrow extension point here and fail loudly until calibrated.
        raise NotImplementedError("ONNX postprocess calibration is required for the exported YOLO model")
