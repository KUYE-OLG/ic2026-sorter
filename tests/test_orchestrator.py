from sorter_control.config import MappingRule, SorterConfig
from sorter_control.orchestrator import SorterOrchestrator
from sorter_control.testing.fakes import FakeCamera, FakeDisplay, FakeMotionController, FakeVisionEngine


def test_orchestrator_processes_one_item_end_to_end():
    config = SorterConfig(mapping_rules=[MappingRule(box_id=1, shape="cube", color="red", item_name="red_cube")])
    orchestrator = SorterOrchestrator(
        config=config,
        camera=FakeCamera(),
        vision=FakeVisionEngine(shape="cube", color="red", confidence=0.94),
        motion=FakeMotionController(),
        display=FakeDisplay(),
    )

    result = orchestrator.process_once()

    assert result.box_id == 1
    assert orchestrator.state_machine.stats.total_success == 1
    assert orchestrator.motion.moves == [(1, 1)]
    assert orchestrator.display.last_result.item_name == "red_cube"
