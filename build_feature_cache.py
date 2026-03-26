from __future__ import annotations

import sys


def main() -> None:
    print(
        "build_feature_cache.py is retired. "
        "The active mainline no longer uses raw shard or feature-cache runtimes. "
        "Use split-manifest + gpu_stream training/inference instead.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
