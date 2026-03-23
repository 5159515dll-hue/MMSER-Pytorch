from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> Path:
    """Expose root-level mainline modules through the legacy package name."""

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


_REPO_ROOT = _ensure_repo_root_on_path()
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in __path__:
    __path__.append(_REPO_ROOT_STR)
