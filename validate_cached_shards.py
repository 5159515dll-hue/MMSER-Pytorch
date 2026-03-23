from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    """解析缓存校验命令行参数。"""

    p = argparse.ArgumentParser(description="Validate cached .pt shards (detect corrupt/truncated files)")
    p.add_argument(
        "--cached-dataset",
        type=Path,
        required=True,
        help="Path to cached shard dir (contains N.pt) or a single shard .pt file",
    )
    p.add_argument(
        "--max-show",
        type=int,
        default=20,
        help="Max number of corrupt shards to show",
    )
    return p.parse_args()


def iter_shards(path: Path) -> list[Path]:
    """把目录或单文件输入统一展开成 shard 路径列表。"""

    path = path.expanduser()
    if path.is_file():
        return [path]
    return sorted(path.glob("*.pt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def main() -> None:
    """逐个读取 shard，检查是否损坏或格式异常。"""

    args = parse_args()
    shards = iter_shards(args.cached_dataset)
    if not shards:
        raise FileNotFoundError(f"No .pt shards found under: {args.cached_dataset}")

    ok = 0
    bad: list[tuple[Path, str]] = []

    for p in shards:
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            # Basic sanity: should be dict with samples or list
            if isinstance(obj, dict) and "samples" in obj:
                _ = obj["samples"]
            elif isinstance(obj, list):
                _ = obj
            else:
                raise RuntimeError(f"Unexpected shard format: type={type(obj).__name__}")
            ok += 1
        except Exception as e:
            bad.append((p, f"{type(e).__name__}: {e}"))

    print(f"Shards: {len(shards)} | OK: {ok} | Corrupt: {len(bad)}", flush=True)
    if bad:
        print("-- Corrupt shards:", flush=True)
        for p, err in bad[: max(1, int(args.max_show))]:
            print(f"  - {p} | {err}", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
