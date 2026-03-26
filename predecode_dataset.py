from __future__ import annotations

import sys


def main() -> None:
    print(
        "predecode_dataset.py is retired. "
        "The active mainline is manifest/gpu_stream-only: run build_split_manifest.py, "
        "then prepare_dataset_media.py, then train.py / batch_inference.py.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
