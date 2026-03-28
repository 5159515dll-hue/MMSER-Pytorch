from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def default_hf_cache_dir() -> Path:
    hf_hub_cache = os.environ.get("HF_HUB_CACHE", "").strip()
    if hf_hub_cache:
        return Path(hf_hub_cache).expanduser()

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path(__file__).resolve().parent / ".hf-cache" / "hub"


def hf_offline_requested() -> bool:
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        value = str(os.environ.get(key, "")).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
    return False


def resolve_local_hf_snapshot(repo_id: str, revision: str | None = None) -> Path | None:
    repo_text = str(repo_id).strip()
    if not repo_text:
        return None

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except Exception:
        return None

    cache_dir = default_hf_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=repo_text,
            revision=revision,
            cache_dir=str(cache_dir),
            local_dir=None,
            local_dir_use_symlinks=False,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return None
    except Exception:
        return None
    return Path(path)


def resolve_hf_pretrained_source(model_name: str, *, revision: str | None = None) -> tuple[str, dict[str, Any]]:
    model_text = str(model_name).strip()
    if not model_text:
        raise ValueError("model_name must not be empty")

    explicit_path = Path(model_text).expanduser()
    if explicit_path.exists():
        return str(explicit_path), {"local_files_only": True}

    snapshot_path = resolve_local_hf_snapshot(model_text, revision=revision)
    if snapshot_path is not None:
        return str(snapshot_path), {"local_files_only": True}

    load_kwargs: dict[str, Any] = {"cache_dir": str(default_hf_cache_dir())}
    if revision is not None and str(revision).strip():
        load_kwargs["revision"] = str(revision).strip()
    if hf_offline_requested():
        load_kwargs["local_files_only"] = True
    return model_text, load_kwargs
