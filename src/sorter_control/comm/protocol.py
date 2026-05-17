from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    action: str
    box_id: int
    seq: int


@dataclass(frozen=True)
class Event:
    kind: str
    seq: int
    status: str


def checksum(body: bytes) -> int:
    value = 0x50  # Shared salt avoids accepting raw NMEA frames by accident.
    for byte in body:
        value ^= byte
    return value & 0xFF


def encode_command(command: Command) -> bytes:
    body = f"{command.action},{command.seq},{command.box_id}".encode("ascii")
    return b"$" + body + f"*{checksum(body):02X}\n".encode("ascii")


def parse_event(frame: bytes) -> Event:
    raw = frame.strip()
    if not raw.startswith(b"$") or b"*" not in raw:
        raise ValueError("bad frame format")
    body, received = raw[1:].rsplit(b"*", 1)
    try:
        expected = int(received, 16)
    except ValueError as exc:
        raise ValueError("bad checksum field") from exc
    actual = checksum(body)
    if actual != expected:
        raise ValueError(f"checksum mismatch: expected {expected:02X}, got {actual:02X}")
    fields = body.decode("ascii").split(",")
    if len(fields) != 3:
        raise ValueError("bad event field count")
    kind, seq_text, status = fields
    return Event(kind=kind, seq=int(seq_text), status=status)
