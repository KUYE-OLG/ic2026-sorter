from sorter_control.state_machine import SorterStateMachine, SorterState


def test_nominal_cycle_reaches_displayed_and_updates_count():
    sm = SorterStateMachine()

    sm.on_item_detected()
    assert sm.state == SorterState.CLASSIFYING

    sm.on_classified(box_id=2, item_name="red_cube", confidence=0.97)
    assert sm.state == SorterState.PLACING

    sm.on_place_confirmed()
    assert sm.state == SorterState.DISPLAYING

    sm.on_display_done()
    assert sm.state == SorterState.IDLE
    assert sm.stats.total_success == 1
    assert sm.stats.per_box[2] == 1


def test_low_confidence_classification_enters_error_state():
    sm = SorterStateMachine(min_confidence=0.8)
    sm.on_item_detected()
    sm.on_classified(box_id=1, item_name="unknown", confidence=0.42)

    assert sm.state == SorterState.ERROR
    assert "confidence" in sm.last_error.lower()


def test_action_timeout_is_reported_as_error():
    sm = SorterStateMachine(action_timeout_s=15.0)
    sm.on_item_detected(now=0.0)

    sm.tick(now=16.0)

    assert sm.state == SorterState.ERROR
    assert "timeout" in sm.last_error.lower()
