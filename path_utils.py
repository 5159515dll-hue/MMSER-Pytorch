from __future__ import annotations

from pathlib import Path


def default_databases_dir(repo_root: Path) -> Path:
    """Return the preferred databases directory for the repository.

    Search order:
    1. `<repo_root>/databases`
    2. `<repo_root>/../databases`

    If neither exists yet, keep the in-repo path as the default creation target.
    """

    root = Path(repo_root).resolve()
    candidates = [root / "databases", root.parent / "databases"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_xlsx_path(repo_root: Path, *filenames: str) -> Path:
    """Return the preferred XLSX metadata path under the detected databases dir."""

    names = filenames or ("video_databases.xlsx",)
    data_root = default_databases_dir(repo_root)
    for name in names:
        candidate = data_root / name
        if candidate.exists():
            return candidate
    return data_root / names[0]
