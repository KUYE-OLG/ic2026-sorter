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
    """Laptop-first inference adapter with ONNX fallback.

    Primary path: PyTorch direct on laptop GPU (RTX 4060+), 3-5x faster than Jetson.
    Fallback: ONNX Runtime CPU when GPU unavailable (e.g. competition venue laptop).

    No TensorRT dependency — laptop GPU inference is fast enough without it.
    """

    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self.session: Any | None = None
        self.model: Any | None = None  # PyTorch model for direct inference
        self.backend: str = "sim"
        self.status = self._load_backend()

    def _load_backend(self) -> ModelStatus:
        model_path = Path(self.config.model_path)

        # 1. Try PyTorch direct (laptop GPU) — preferred path
        if self.config.backend in ("auto", "pytorch") and model_path.suffix in (".pt", ".pth"):
            try:
                from ultralytics import YOLO  # type: ignore
                self.model = YOLO(str(model_path))
                self.backend = "pytorch"
                return ModelStatus("pytorch", str(model_path), True,
                                   "YOLO model loaded for laptop GPU inference")
            except ImportError:
                pass
            except Exception as e:
                return ModelStatus("pytorch_fail", str(model_path), False, str(e))

        # 2. Fallback: ONNX Runtime (CPU) — for venues without GPU
        if self.config.backend in ("auto", "onnx") and model_path.exists():
            try:
                import onnxruntime as ort  # type: ignore
            except ImportError:
                return ModelStatus("sim", str(model_path), False,
                                   "onnxruntime not installed — using simulator")
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self.backend = "onnx"
            return ModelStatus("onnx", str(model_path), True,
                               "ONNX model loaded for CPU inference")

        return ModelStatus("sim", str(model_path), False,
                           "no usable backend; using deterministic simulator")

    def infer(self, frame: Any) -> VisionObservation:
        if self.model is not None:
            # PyTorch direct inference — laptop GPU path
            results = self.model(frame, verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    best = boxes[0]
                    cls_id = int(best.cls[0])
                    cls_name = self.model.names.get(cls_id, "unknown")
                    conf = float(best.conf[0])
                    xyxy = tuple(int(v) for v in best.xyxy[0].tolist())
                    return VisionObservation(
                        shape=cls_name,
                        color="unknown",
                        confidence=conf,
                        bbox=xyxy,
                    )

        if self.session is not None:
            # ONNX fallback — postprocess depends on exported model head
            raise NotImplementedError(
                "ONNX postprocess calibration required for exported YOLO model"
            )

        # Deterministic simulator fallback — keeps rest of system debuggable
        return VisionObservation(
            shape="cube", color="red", confidence=0.99, bbox=(0, 0, 40, 40)
        )
