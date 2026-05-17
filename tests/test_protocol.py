from sorter_control.comm.protocol import Command, Event, encode_command, parse_event


def test_command_frame_has_checksum_and_newline():
    frame = encode_command(Command(action="MOVE", box_id=3, seq=12))

    assert frame.startswith(b"$")
    assert frame.endswith(b"\n")
    assert b"MOVE" in frame
    assert b"*" in frame


def test_parse_valid_ack_event():
    frame = b"$ACK,12,OK*1E\n"

    event = parse_event(frame)

    assert event == Event(kind="ACK", seq=12, status="OK")


def test_parse_event_rejects_bad_checksum():
    try:
        parse_event(b"$ACK,12,OK*00\n")
    except ValueError as exc:
        assert "checksum" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")
