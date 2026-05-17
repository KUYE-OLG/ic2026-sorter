from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sorter_control.config import SorterConfig


@dataclass(frozen=True)
class VisionObservation:
    shape: str | None = None
    color: str | None = None
    text: str | None = None
    qr_payload: str | None = None
    image_mark: str | None = None
    defect: str | None = None
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    center_mm: tuple[float, float] | None = None
    crop: Any | None = None


@dataclass(frozen=True)
class SortDecision:
    seq: int
    box_id: int
    item_name: str
    confidence: float
    shape: str | None = None
    color: str | None = None
    text: str | None = None
    qr_payload: str | None = None
    image_mark: str | None = None
    defect: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    center_mm: tuple[float, float] | None = None
    display_image_path: str | None = None


class DecisionFusion:
    def __init__(self, config: SorterConfig, run_dir: str | Path = "runs/current") -> None:
        self.config = config
        self.run_dir = Path(run_dir)

    def decide(self, observation: VisionObservation, seq: int) -> SortDecision:
        candidates = [rule for rule in self.config.mapping_rules if rule.matches(observation)]
        if not candidates:
            raise ValueError(
                "no mapping rule for observation "
                f"shape={observation.shape!r}, color={observation.color!r}, "
                f"text={observation.text!r}, qr={observation.qr_payload!r}"
            )
        rule = max(candidates, key=lambda r: r.priority)
        return SortDecision(
            seq=seq,
            box_id=rule.box_id,
            item_name=rule.item_name,
            confidence=observation.confidence,
            shape=observation.shape,
            color=observation.color,
            text=observation.text,
            qr_payload=observation.qr_payload,
            image_mark=observation.image_mark,
            defect=observation.defect,
            bbox=observation.bbox,
            center_mm=observation.center_mm,
            display_image_path=str(self.run_dir / f"item_{seq:03d}.jpg"),
        )
