from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


DEFAULT_MODELS: list[tuple[str, str | None]] = [
    ("FacebookAI/xlm-roberta-large", None),
    ("MCG-NJU/videomae-large", None),
    ("microsoft/wavlm-large", "e4e472c491084b2c6fb9736099130aa805159c62"),
]


def _default_cache_dir() -> Path:
    hf_hub_cache = os.environ.get("HF_HUB_CACHE", "").strip()
    if hf_hub_cache:
        return Path(hf_hub_cache).expanduser()

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path(__file__).resolve().parent / ".hf-cache" / "hub"


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
        default=_default_cache_dir(),
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
    cache_dir = args.cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for repo_id, revision in DEFAULT_MODELS:
        cache_dir_str = str(cache_dir)
        if not args.force:
            try:
                path = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    cache_dir=cache_dir_str,
                    local_dir=None,
                    local_dir_use_symlinks=False,
                    local_files_only=True,
                )
                print(f"cached: {repo_id} -> {path}")
                continue
            except LocalEntryNotFoundError:
                pass

        print(f"downloading: {repo_id} revision={revision} cache_dir={cache_dir}")
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir_str,
            local_dir=None,
            local_dir_use_symlinks=False,
            resume_download=(not bool(args.force)),
            force_download=bool(args.force),
        )
        print(f"done: {repo_id} -> {path}")


if __name__ == "__main__":
    main()
