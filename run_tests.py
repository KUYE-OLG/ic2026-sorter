#!/usr/bin/env python3
"""Tiny test runner for environments without pytest."""
from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    passed = 0
    failed = 0
    for path in tests:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        try:
            spec.loader.exec_module(module)
        except Exception:
            failed += 1
            print(f"ERROR {path.name} import")
            traceback.print_exc()
            continue
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            try:
                fn()
            except Exception:
                failed += 1
                print(f"FAIL {path.name}::{name}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"PASS {path.name}::{name}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
