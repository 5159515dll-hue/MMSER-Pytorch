from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


DEFAULT_MODELS: list[tuple[str, str | None]] = [
    ("FacebookAI/xlm-roberta-large", None),
    ("MCG-NJU/videomae-large", None),
    ("microsoft/wavlm-large", "e4e472c491084b2c6fb9736099130aa805159c62"),
]

TEXT_MODEL_REPOS = {"FacebookAI/xlm-roberta-large"}


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


def _verify_safetensors_file(path: Path) -> None:
    with safe_open(str(path), framework="pt") as handle:
        # Force safetensors to parse the file header and key index.
        _ = list(handle.keys())


def _verify_snapshot_weights(snapshot_path: Path) -> None:
    single_file = snapshot_path / "model.safetensors"
    shard_index = snapshot_path / "model.safetensors.index.json"

    if single_file.is_file():
        _verify_safetensors_file(single_file)
        return

    if shard_index.is_file():
        payload = json.loads(shard_index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        shard_names = sorted(set(str(v) for v in weight_map.values()))
        if not shard_names:
            raise RuntimeError(f"empty weight_map in {shard_index}")
        for shard_name in shard_names:
            shard_path = snapshot_path / shard_name
            if not shard_path.is_file():
                raise RuntimeError(f"missing shard file: {shard_path}")
            _verify_safetensors_file(shard_path)
        return

    raise RuntimeError(f"no safetensors weights found in {snapshot_path}")


def _verify_cached_snapshot(repo_id: str, revision: str | None, snapshot_path: Path) -> None:
    AutoConfig.from_pretrained(str(snapshot_path), revision=revision, local_files_only=True)
    _verify_snapshot_weights(snapshot_path)
    if repo_id in TEXT_MODEL_REPOS:
        AutoTokenizer.from_pretrained(str(snapshot_path), revision=revision, local_files_only=True)


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
                _verify_cached_snapshot(repo_id=repo_id, revision=revision, snapshot_path=Path(path))
                print(f"cached+verified: {repo_id} -> {path}")
                continue
            except LocalEntryNotFoundError:
                pass
            except Exception as exc:
                print(f"cache-invalid: {repo_id} -> {type(exc).__name__}: {exc}")

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
        _verify_cached_snapshot(repo_id=repo_id, revision=revision, snapshot_path=Path(path))
        print(f"done: {repo_id} -> {path}")


if __name__ == "__main__":
    main()
