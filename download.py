from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODELS: list[tuple[str, str | None]] = [
    ("FacebookAI/xlm-roberta-large", None),
    ("MCG-NJU/videomae-large", None),
    ("microsoft/wavlm-large", "e4e472c491084b2c6fb9736099130aa805159c62"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download the verified Hugging Face model snapshots used by the MELD "
            "benchmarks into the local HF cache."
        )
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory passed to huggingface_hub snapshot_download.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh download even if the snapshot already exists in the local cache.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for repo_id, revision in DEFAULT_MODELS:
        print(f"downloading: {repo_id} revision={revision}")
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(args.cache_dir.expanduser()) if args.cache_dir is not None else None,
            local_dir=None,
            local_dir_use_symlinks=False,
            resume_download=(not bool(args.force)),
            force_download=bool(args.force),
        )
        print(f"done: {repo_id} -> {path}")


if __name__ == "__main__":
    main()
