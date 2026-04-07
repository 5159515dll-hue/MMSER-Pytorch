from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_store import validate_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a paper-grade closed-loop manifest-driven run directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_run_dir(args.run_dir.expanduser())
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary.get("issues"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
