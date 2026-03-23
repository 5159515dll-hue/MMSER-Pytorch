from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from motion_prosody.predecode_motion_audio import *  # noqa: F401,F403
from motion_prosody.predecode_motion_audio import main


if __name__ == "__main__":
    main()
