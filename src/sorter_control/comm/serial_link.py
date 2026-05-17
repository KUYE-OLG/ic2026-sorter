from __future__ import annotations

import time
from typing import Any

from sorter_control.comm.protocol import Command, Event, encode_command, parse_event
from sorter_control.config import MotionConfig


class SerialMotionController:
    def __init__(self, config: MotionConfig) -> None:
        self.config = config
        self._serial: Any | None = None

    def open(self) -> None:
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyserial is required for motion control") from exc
        self._serial = serial.Serial(self.config.serial_port, self.config.baudrate, timeout=0.1)

    def move_to_box(self, box_id: int, seq: int) -> Event:
        if self._serial is None:
            self.open()
        assert self._serial is not None
        self._serial.write(encode_command(Command("MOVE", box_id=box_id, seq=seq)))
        deadline = time.monotonic() + self.config.ack_timeout_s
        while time.monotonic() < deadline:
            line = self._serial.readline()
            if not line:
                continue
            event = parse_event(line)
            if event.seq == seq:
                if event.status != "OK":
                    raise RuntimeError(f"motion controller rejected seq {seq}: {event.status}")
                return event
        raise TimeoutError(f"motion ACK timeout for seq {seq}")
