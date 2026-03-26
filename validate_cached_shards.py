from __future__ import annotations

import sys


def main() -> None:
    print(
        "validate_cached_shards.py is retired with the cached/feature-cache runtime. "
        "The active mainline validates data through split-manifest + prepare_dataset_media.py.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
