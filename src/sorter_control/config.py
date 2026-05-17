from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MappingRule:
    box_id: int
    item_name: str
    shape: str | None = None
    color: str | None = None
    text: str | None = None
    qr_payload: str | None = None
    image_mark: str | None = None
    defect: str | None = None

    def matches(self, observation: Any) -> bool:
        for key in ("qr_payload", "text", "shape", "color", "image_mark", "defect"):
            expected = getattr(self, key)
            if expected is not None and getattr(observation, key, None) != expected:
                return False
        return True

    @property
    def priority(self) -> int:
        # QR/text are explicit task identifiers, so they outrank visual fallback rules.
        if self.qr_payload:
            return 100
        if self.text:
            return 90
        if self.image_mark:
            return 70
        if self.defect:
            return 60
        score = 0
        if self.shape:
            score += 20
        if self.color:
            score += 20
        return score


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    exposure: int | None = None


@dataclass(frozen=True)
class AIConfig:
    backend: str = "auto"
    model_path: str = "models/sorter_yolo.onnx"
    input_size: int = 640
    confidence_threshold: float = 0.6


@dataclass(frozen=True)
class MotionConfig:
    serial_port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    ack_timeout_s: float = 2.0


@dataclass(frozen=True)
class DisplayConfig:
    fullscreen: bool = False
    width: int = 1024
    height: int = 600


@dataclass(frozen=True)
class SorterConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    mapping_rules: list[MappingRule] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SorterConfig":
        return SorterConfig(
            camera=CameraConfig(**data.get("camera", {})),
            ai=AIConfig(**data.get("ai", {})),
            motion=MotionConfig(**data.get("motion", {})),
            display=DisplayConfig(**data.get("display", {})),
            mapping_rules=[MappingRule(**item) for item in data.get("mapping_rules", [])],
        )

    @staticmethod
    def load(path: str | Path) -> "SorterConfig":
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML config files") from exc
        with Path(path).open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        return SorterConfig.from_dict(data)
