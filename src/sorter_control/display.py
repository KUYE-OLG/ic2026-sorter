from __future__ import annotations

from sorter_control.config import DisplayConfig
from sorter_control.vision.classifier import SortDecision


class ConsoleDisplay:
    def __init__(self, config: DisplayConfig) -> None:
        self.config = config
        self.last_result: SortDecision | None = None

    def show_result(self, result: SortDecision, total_success: int) -> None:
        self.last_result = result
        print(
            f"SEQ={result.seq:03d} ITEM={result.item_name} BOX={result.box_id} "
            f"CONF={result.confidence:.2f} TOTAL={total_success} IMAGE={result.display_image_path}"
        )
