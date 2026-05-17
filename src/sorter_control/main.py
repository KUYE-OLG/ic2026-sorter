from __future__ import annotations

import argparse

from sorter_control.camera import CameraCapture
from sorter_control.comm.serial_link import SerialMotionController
from sorter_control.config import SorterConfig
from sorter_control.display import ConsoleDisplay
from sorter_control.orchestrator import SorterOrchestrator
from sorter_control.vision.engine import VisionEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IC2026 intelligent sorter control loop")
    parser.add_argument("--config", default="configs/task_config.yaml")
    parser.add_argument("--once", action="store_true", help="process one item and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = SorterConfig.load(args.config)
    orchestrator = SorterOrchestrator(
        config=config,
        camera=CameraCapture(config.camera),
        vision=VisionEngine(config.ai),
        motion=SerialMotionController(config.motion),
        display=ConsoleDisplay(config.display),
    )
    while True:
        orchestrator.process_once()
        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
