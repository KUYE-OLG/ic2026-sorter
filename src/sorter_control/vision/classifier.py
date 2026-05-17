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
    """Fuses YOLO detection with geometry-based fallback for unknown shapes.

    Strategy:
    1. YOLO high confidence (>0.85) + geometry consistent → trust YOLO
    2. Otherwise → fall back to geometry template matching
    """

    def __init__(self, config: SorterConfig, run_dir: str | Path = "runs/current") -> None:
        self.config = config
        self.run_dir = Path(run_dir)

    def decide(self, observation: VisionObservation, seq: int) -> SortDecision:
        # If YOLO confidence is low or no shape, try geometry fallback
        if observation.shape is None or observation.confidence < 0.85:
            observation = self._geometry_fallback(observation)

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

    def _geometry_fallback(self, observation: VisionObservation) -> VisionObservation:
        """Try OpenCV geometry classifier when YOLO is uncertain."""
        from sorter_control.vision.geometry_classifier import (
            extract_geometry,
            match_by_geometry,
        )
        import numpy as np

        if observation.crop is None:
            return observation  # No image crop to analyze

        # Convert crop to contour
        img = np.asarray(observation.crop)
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2).astype(np.uint8)
        else:
            gray = img.astype(np.uint8)

        # Find largest contour
        import cv2
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return observation

        largest = max(contours, key=cv2.contourArea)
        features = extract_geometry(largest)
        shape_name, geo_conf = match_by_geometry(features)

        if geo_conf > 0.5:  # At least 3.5/7 match
            return VisionObservation(
                shape=shape_name,
                color=observation.color,
                text=observation.text,
                qr_payload=observation.qr_payload,
                image_mark=observation.image_mark,
                defect=observation.defect,
                confidence=geo_conf,
                bbox=observation.bbox,
                center_mm=observation.center_mm,
                crop=observation.crop,
            )

        return observation
