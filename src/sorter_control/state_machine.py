from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class SorterState(Enum):
    IDLE = auto()
    CAPTURING = auto()
    CLASSIFYING = auto()
    PLACING = auto()
    DISPLAYING = auto()
    ERROR = auto()


@dataclass
class SorterStats:
    total_success: int = 0
    total_error: int = 0
    per_box: dict[int, int] = field(default_factory=dict)


class SorterStateMachine:
    def __init__(self, min_confidence: float = 0.6, action_timeout_s: float = 15.0) -> None:
        self.state = SorterState.IDLE
        self.min_confidence = min_confidence
        self.action_timeout_s = action_timeout_s
        self.stats = SorterStats()
        self.last_error = ""
        self._state_started_at = time.monotonic()
        self._pending_box_id: int | None = None
        self._pending_item_name = ""

    def _transition(self, state: SorterState, now: float | None = None) -> None:
        self.state = state
        self._state_started_at = time.monotonic() if now is None else now

    def on_item_detected(self, now: float | None = None) -> None:
        if self.state not in (SorterState.IDLE, SorterState.CAPTURING):
            self._fail(f"item detected in invalid state {self.state.name}", now)
            return
        self._transition(SorterState.CLASSIFYING, now)

    def on_classified(self, box_id: int, item_name: str, confidence: float, now: float | None = None) -> None:
        if self.state != SorterState.CLASSIFYING:
            self._fail(f"classification in invalid state {self.state.name}", now)
            return
        if confidence < self.min_confidence:
            self._fail(f"classification confidence {confidence:.2f} below threshold {self.min_confidence:.2f}", now)
            return
        self._pending_box_id = box_id
        self._pending_item_name = item_name
        self._transition(SorterState.PLACING, now)

    def on_place_confirmed(self, now: float | None = None) -> None:
        if self.state != SorterState.PLACING:
            self._fail(f"place confirmation in invalid state {self.state.name}", now)
            return
        if self._pending_box_id is None:
            self._fail("place confirmation without pending box", now)
            return
        self.stats.total_success += 1
        self.stats.per_box[self._pending_box_id] = self.stats.per_box.get(self._pending_box_id, 0) + 1
        self._transition(SorterState.DISPLAYING, now)

    def on_display_done(self, now: float | None = None) -> None:
        if self.state != SorterState.DISPLAYING:
            self._fail(f"display done in invalid state {self.state.name}", now)
            return
        self._pending_box_id = None
        self._pending_item_name = ""
        self._transition(SorterState.IDLE, now)

    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.state in (SorterState.IDLE, SorterState.ERROR):
            return
        if now - self._state_started_at > self.action_timeout_s:
            self._fail(f"timeout in state {self.state.name}", now)

    def reset_error(self, now: float | None = None) -> None:
        if self.state != SorterState.ERROR:
            return
        self.last_error = ""
        self._transition(SorterState.IDLE, now)

    def _fail(self, message: str, now: float | None = None) -> None:
        self.last_error = message
        self.stats.total_error += 1
        self._transition(SorterState.ERROR, now)
