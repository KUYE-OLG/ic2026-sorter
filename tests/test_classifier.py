from sorter_control.config import MappingRule, SorterConfig
from sorter_control.vision.classifier import DecisionFusion, VisionObservation


def test_decision_fusion_prefers_qr_payload_mapping_over_shape_color():
    config = SorterConfig(
        mapping_rules=[
            MappingRule(box_id=1, shape="cube", color="red", item_name="red_cube"),
            MappingRule(box_id=4, qr_payload="BOX-4", item_name="qr_target"),
        ]
    )
    fusion = DecisionFusion(config)
    obs = VisionObservation(
        shape="cube",
        color="red",
        qr_payload="BOX-4",
        confidence=0.91,
        bbox=(10, 20, 40, 50),
        crop=None,
    )

    result = fusion.decide(obs, seq=7)

    assert result.box_id == 4
    assert result.item_name == "qr_target"
    assert result.confidence == 0.91
    assert result.seq == 7


def test_decision_fusion_falls_back_to_shape_color_rule():
    config = SorterConfig(mapping_rules=[MappingRule(box_id=2, shape="cylinder", color="blue", item_name="blue_cylinder")])
    result = DecisionFusion(config).decide(VisionObservation(shape="cylinder", color="blue", confidence=0.86), seq=3)

    assert result.box_id == 2
    assert result.item_name == "blue_cylinder"


def test_decision_fusion_raises_for_unmapped_item():
    config = SorterConfig(mapping_rules=[MappingRule(box_id=1, shape="cube", color="red", item_name="red_cube")])
    fusion = DecisionFusion(config)

    try:
        fusion.decide(VisionObservation(shape="sphere", color="green", confidence=0.9), seq=1)
    except ValueError as exc:
        assert "no mapping" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")
