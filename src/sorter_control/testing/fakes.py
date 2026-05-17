from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sorter_control.vision.classifier import SortDecision, VisionObservation


class FakeCamera:
    def read(self) -> dict[str, str]:
        return {"frame": "synthetic"}


@dataclass
class FakeVisionEngine:
    shape: str = "cube"
    color: str = "red"
    confidence: float = 0.95
    qr_payload: str | None = None
    text: str | None = None

    def infer(self, frame: Any) -> VisionObservation:
        return VisionObservation(
            shape=self.shape,
            color=self.color,
            qr_payload=self.qr_payload,
            text=self.text,
            confidence=self.confidence,
        )


class FakeMotionController:
    def __init__(self) -> None:
        self.moves: list[tuple[int, int]] = []

    def move_to_box(self, box_id: int, seq: int) -> None:
        self.moves.append((box_id, seq))


class FakeDisplay:
    def __init__(self) -> None:
        self.last_result: SortDecision | None = None

    def show_result(self, result: SortDecision, total_success: int) -> None:
        self.last_result = result
