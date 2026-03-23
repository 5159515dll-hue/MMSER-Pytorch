from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transition shim for the archived V1 single-sample inference entrypoint"
    )
    parser.add_argument(
        "--legacy-path",
        action="store_true",
        help="Print the archived V1 inference script path and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy_script = Path("legacy/baseline_v1/inference.py")
    message = (
        "Root-level inference.py is now a transition shim.\n"
        "The active project focus is the manifest-driven motion_prosody pipeline, "
        "which is maintained through batch_inference.py.\n"
        f"For the archived V1 single-sample path, use: python {legacy_script}"
    )
    stream = sys.stdout if args.legacy_path else sys.stderr
    print(message, file=stream)
    if not args.legacy_path:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
