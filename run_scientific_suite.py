from __future__ import annotations

import sys


def main() -> None:
    print(
        "run_scientific_suite.py is retired from the active mainline. "
        "Its old workflow depended on cached/feature-cache execution paths that are no longer supported.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
