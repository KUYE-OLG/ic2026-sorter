from __future__ import annotations

from typing import Protocol, Any

from sorter_control.config import SorterConfig
from sorter_control.state_machine import SorterStateMachine
from sorter_control.vision.classifier import DecisionFusion, SortDecision, VisionObservation


class CameraLike(Protocol):
    def read(self) -> Any: ...


class VisionLike(Protocol):
    def infer(self, frame: Any) -> VisionObservation: ...


class MotionLike(Protocol):
    def move_to_box(self, box_id: int, seq: int) -> Any: ...


class DisplayLike(Protocol):
    def show_result(self, result: SortDecision, total_success: int) -> None: ...


class SorterOrchestrator:
    def __init__(
        self,
        config: SorterConfig,
        camera: CameraLike,
        vision: VisionLike,
        motion: MotionLike,
        display: DisplayLike,
        state_machine: SorterStateMachine | None = None,
    ) -> None:
        self.config = config
        self.camera = camera
        self.vision = vision
        self.motion = motion
        self.display = display
        self.state_machine = state_machine or SorterStateMachine(min_confidence=config.ai.confidence_threshold)
        self.fusion = DecisionFusion(config)
        self.seq = 0

    def process_once(self) -> SortDecision:
        self.seq += 1
        frame = self.camera.read()
        self.state_machine.on_item_detected()
        observation = self.vision.infer(frame)
        result = self.fusion.decide(observation, self.seq)
        self.state_machine.on_classified(result.box_id, result.item_name, result.confidence)
        self.motion.move_to_box(result.box_id, result.seq)
        self.state_machine.on_place_confirmed()
        self.display.show_result(result, self.state_machine.stats.total_success)
        self.state_machine.on_display_done()
        return result
